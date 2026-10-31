import os
import re
import json
from urllib.parse import urlparse

import requests
from typing import Annotated, TypedDict

from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Add OPENAI_API_KEY to your .env file"

console = Console()
MODEL = "gpt-6-luna"
REQUEST_TIMEOUT = (5, 20)
MAX_PAGE_CHARS = 6000
step = 0
show_details = False
trace_log: list[dict[str, str]] = []
conversation_messages: list[AnyMessage] = []
tool_cache: dict[str, str] = {}
session_sources: list[dict[str, str]] = []

http = requests.Session()
http.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchAgent/1.0; "
        "+https://www.google.com/webmasters/robot.html)"
    ),
    "Accept": "text/html,application/xhtml+xml",
})


# ---------- Tools ----------
@tool
def web_search(query: str) -> str:
    """Search the web. Use for facts, current events, distances, prices, or anything you are unsure about.
    Returns titles, links and short snippets."""
    query = query.strip()
    if not query:
        return "Search failed: query cannot be empty."
    try:
        results = DDGS().text(query, max_results=6)
    except Exception as e:
        return f"Search failed ({type(e).__name__}): {e}"
    if not results:
        return "No results found."
    lines = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled result").strip()
        url = result.get("href", "").strip()
        snippet = result.get("body", "No snippet available").strip()
        lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
    return "\n".join(lines)


@tool
def scrape_page(url: str) -> str:
    """Fetch a web page and return structured, readable text.

    Use after web_search to read a result in detail. The extractor prefers
    article/main content, removes navigation and boilerplate, and includes
    page metadata so the model can judge source quality.
    """
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Could not open page: URL must be an absolute http(s) URL."

    try:
        response = http.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and "text/" not in content_type:
            return f"Could not open page: unsupported content type ({content_type or 'unknown'})."

        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"
        description_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        description = description_tag.get("content", "").strip() if description_tag else ""

        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "form", "noscript", "svg", "iframe"]):
            tag.decompose()

        content = (
            soup.find("article")
            or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.body
            or soup
        )
        paragraphs = []
        for element in content.find_all(["h1", "h2", "h3", "p", "li"]):
            line = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if line and len(line) > 20 and line not in paragraphs:
                paragraphs.append(line)
        text = "\n\n".join(paragraphs)
        if not text:
            text = re.sub(r"\s+", " ", content.get_text(" ", strip=True))
        if not text:
            return "Page had no readable text."

        metadata = [
            f"TITLE: {title}",
            f"SOURCE: {response.url}",
        ]
        if description:
            metadata.append(f"DESCRIPTION: {description}")
        return "\n".join(metadata) + "\n\nCONTENT:\n" + text[:MAX_PAGE_CHARS]
    except requests.RequestException as e:
        return f"Could not open page ({type(e).__name__}): {e}"
    except Exception as e:
        return f"Could not parse page ({type(e).__name__}): {e}"


tools_by_name = {t.name: t for t in [web_search, scrape_page]}
llm = ChatOpenAI(model=MODEL).bind_tools(list(tools_by_name.values()))

SYSTEM = SystemMessage(content=(
    "You are a helpful research agent. If a question needs facts you are not certain of, "
    "or needs current information, use web_search first. Use scrape_page to read a promising "
    "result in detail. Give a short, clear final answer.\n\n"
    "You have memory for the entire current chat session. Before calling a tool, inspect "
    "previous questions, tool results, and answers. Reuse earlier findings when they answer "
    "the current question. Do not call a tool again for the same request or source unless "
    "the user asks for an update, the information is time-sensitive, the earlier result is "
    "insufficient, or a different source is needed. Mention when you rely on earlier findings. "
    "A reset starts a new memory context.\n\n"
    "When you use web_search or scrape_page findings, cite them in the answer using "
    "numbered inline citations such as [1] and [2]. End with a Markdown heading exactly "
    "'## Sources' followed by a numbered list of the relevant source URLs. Do not invent "
    "URLs; use only URLs present in the tool results."
))


# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def text_of(content) -> str:
    """Some models return content as a list of blocks. Keep only the text."""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )


def clean_markdown(content: str) -> str:
    """Keep Markdown readable in a terminal while preserving paragraph breaks."""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"[ \t]+\n", "\n", content)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


def print_markdown(content: str, *, title: str, border_style: str = "green") -> None:
    """Render Markdown inside a padded panel sized for the current terminal."""
    markdown = Markdown(clean_markdown(content), code_theme="monokai")
    console.print(Panel(
        Padding(markdown, (0, 1)),
        title=title,
        border_style=border_style,
        expand=False,
        padding=(1, 1),
        width=min(console.width, 110),
    ))


def compact_text(content: str, limit: int = 180) -> str:
    one_line = re.sub(r"\s+", " ", content).strip()
    return one_line if len(one_line) <= limit else one_line[:limit - 1].rstrip() + "…"


def cache_key(tool_name: str, arguments: dict) -> str:
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"


def remember_sources(output: str) -> None:
    """Collect source URLs from live or cached web-tool output for final citations."""
    for url in re.findall(r"https?://[^\s)\]>]+", output):
        url = url.rstrip(".,;")
        title_match = re.search(
            rf"(?:TITLE:\s*|^\s*\d+\.\s*)([^\n]+).*?{re.escape(url)}",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        title = title_match.group(1).strip() if title_match else urlparse(url).netloc
        if not any(source["url"] == url for source in session_sources):
            session_sources.append({"title": title, "url": url})


def add_source_citations(answer: str) -> str:
    """Guarantee a readable Markdown Sources section for web-backed answers."""
    answer = clean_markdown(answer)
    if not session_sources:
        return answer

    citations = ", ".join(
        f"[{index}]" for index in range(1, len(session_sources) + 1)
    )
    if not re.search(r"\[\d+\]", answer):
        answer += f"\n\n_Web-backed claims: {citations}_"

    if "## Sources" not in answer:
        answer += "\n\n## Sources\n"
        answer += "\n".join(
            f"{index}. [{source['title']}]({source['url']})"
            for index, source in enumerate(session_sources, 1)
        )
    return answer


def show_trace(*, expanded: bool | None = None) -> None:
    """Show the current run's captured activity, compact or expanded."""
    if not trace_log:
        console.print("[dim]No activity has been captured yet.[/dim]")
        return
    expanded = show_details if expanded is None else expanded
    console.print(Rule("Run trace", style="cyan"))
    for item in trace_log:
        if item["kind"] == "thinking":
            console.print(f"[bold cyan]Step {item['step']}[/bold cyan] [dim]LLM[/dim]")
            if expanded:
                console.print(Padding(Markdown(item["detail"]), (0, 2, 1, 2)))
        else:
            title = item["name"]
            if expanded:
                print_markdown(item["detail"], title=title, border_style="magenta")
            else:
                console.print(
                    f"  [magenta]• {title}[/magenta] "
                    f"[dim]{item['size']} chars · {compact_text(item['detail'])}[/dim]"
                )


# ---------- Nodes ----------
def llm_node(state: State):
    global step
    step += 1
    trace_log.append({
        "kind": "thinking",
        "step": str(step),
        "detail": "The model is deciding whether it needs a search, scraper, or calculator.",
    })
    console.print(
        f"\n[bold cyan]Step {step} · thinking[/bold cyan] "
        "[dim](details available with 'trace')[/dim] 🧠"
    )
    with console.status("[cyan]Reasoning…[/cyan]", spinner="dots"):
        response = llm.invoke([SYSTEM] + state["messages"])
    if response.tool_calls:
        for call in response.tool_calls:
            console.print(
                f"  [yellow]→ {call['name']}[/yellow] "
                f"[dim]{compact_text(str(call['args']), 100)}[/dim]"
            )
    else:
        console.print("  [green]→ ready to answer[/green]")
    return {"messages": [response]}


def tool_node(state: State):
    console.print("[bold magenta]🛠  Running tool(s)[/bold magenta]")
    results = []
    for call in state["messages"][-1].tool_calls:
        tool_name = call["name"]
        key = cache_key(tool_name, call["args"])
        cached = key in tool_cache
        if cached:
            output = tool_cache[key]
        else:
            with console.status(f"[magenta]Running {tool_name}…[/magenta]", spinner="dots"):
                output = str(tools_by_name[tool_name].invoke(call["args"]))
            tool_cache[key] = output
        remember_sources(output)
        trace_log.append({
            "kind": "tool",
            "name": f"{tool_name} ({'memory' if cached else 'live'})",
            "detail": output,
            "size": str(len(output)),
        })
        if show_details:
            print_markdown(
                output,
                title=f"[bold]{tool_name} · {'memory' if cached else 'live'}[/bold]",
                border_style="magenta",
            )
        else:
            console.print(
                f"  [magenta]✓ {tool_name}[/magenta] "
                f"[dim]{'reused from memory' if cached else 'fresh result'} · "
                f"{len(output):,} chars · collapsed "
                f"(use 'trace' to expand)[/dim]"
            )
        results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {"messages": results}


# ---------- Routing ----------
def route(state: State):
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", route, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")
graph = builder.compile()


# ---------- Live UI ----------
def show_graph():
    diagram = Text(
        "START ─→ [llm] ──needs a tool?──→ [tools]\n"
        "            ↑                        │\n"
        "            └────── result ───────────┘\n"
        "            └── no tool needed ──→ END"
    )
    console.print(Panel(diagram, title="[bold]Agent graph[/bold]", border_style="cyan"))


def show_help() -> None:
    table = Table(title="Commands", border_style="cyan", show_header=False)
    table.add_column("Command", style="bold green")
    table.add_column("Action", style="white")
    table.add_row("quit / exit", "Close the assistant")
    table.add_row("help", "Show this command list")
    table.add_row("trace", "Show this run's thinking and tool activity")
    table.add_row("verbose", "Expand tool output while the agent runs")
    table.add_row("compact", "Collapse tool output while the agent runs")
    table.add_row("reset", "Forget this session's conversation and tool cache")
    console.print(table)


def main():
    global conversation_messages, session_sources, show_details, step, tool_cache, trace_log
    console.print(Panel(
        "[bold]Research Agent[/bold]\n"
        "Ask a question and the agent will search, inspect pages, and explain its findings.",
        border_style="green",
    ))
    show_graph()
    show_help()

    while True:
        question = console.input("\n[bold green]Ask the agent[/bold green] (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question.lower() == "help":
            show_help()
            continue
        if question.lower() == "trace":
            show_trace(expanded=True)
            continue
        if question.lower() == "verbose":
            show_details = True
            console.print("[green]Verbose tool output enabled.[/green]")
            continue
        if question.lower() == "compact":
            show_details = False
            console.print("[green]Compact tool output enabled.[/green]")
            continue
        if question.lower() == "reset":
            conversation_messages = []
            session_sources = []
            tool_cache = {}
            trace_log = []
            console.print("[green]Session memory and tool cache cleared.[/green]")
            continue
        if not question:
            continue

        step = 0
        trace_log = []
        console.print(Rule(f"[bold]{question}[/bold]", style="green"))
        try:
            result = graph.invoke(
                {"messages": conversation_messages + [HumanMessage(content=question)]},
                config={"recursion_limit": 12},
            )
            conversation_messages = result["messages"]
            answer = add_source_citations(text_of(result["messages"][-1].content))
            print_markdown(answer, title="✅ Final answer", border_style="green")
            console.print("[dim]Tip: type 'trace' to expand this run's thinking and tools.[/dim]")
        except Exception as e:
            console.print(Panel(str(e), title="Error", border_style="red"))


if __name__ == "__main__":
    main()