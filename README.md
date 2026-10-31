# Research Agent

A terminal-based AI research assistant built with LangGraph, LangChain, Rich, and OpenAI.

It can:

- Search the web for current information.
- Read and extract useful content from web pages.
- Remember previous questions and answers during one session.
- Reuse identical search results without making the same request again.
- Display readable Markdown directly in the terminal.
- Include source citations and URLs in web-based answers.

## Requirements

- Python 3.12 or newer
- An OpenAI API key

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```env
OPENAI_API_KEY=your_openai_api_key
```

## Run

```bash
python3 main.py
```

Ask questions at the prompt. For example:

```text
Who is Ada Lovelace?
```

When web research is used, the final answer includes Markdown citations and a
`Sources` section with clickable URLs.

## Commands

| Command | Description |
| --- | --- |
| `help` | Show available commands |
| `trace` | Show the current run's thinking and tool activity |
| `verbose` | Display tool results immediately |
| `compact` | Collapse tool results while the agent runs |
| `reset` | Clear conversation memory, cached tools, and sources |
| `export` | Export the architecture as editable Mermaid and SVG files |
| `quit` or `exit` | Close the application |

## Session memory

The agent remembers the conversation until the application is closed or you
use `reset`.

It also caches identical search and page-reading requests during the session.
This prevents repeated questions from making unnecessary web requests. The
agent can still search again when information is current, time-sensitive, or
the previous result is not enough.

## Project files

- `main.py` - Application code and terminal interface
- `requirements.txt` - Python dependencies
- `.env` - Local API key configuration; do not commit this file

## Export the architecture

Start the application and enter:

```text
export
```

This creates:

- `architecture.mmd` - editable Mermaid diagram source
- `architecture.svg` - image version for browsers, presentations, and documentation

The SVG export uses Graphviz's `dot` command when it is installed. The Mermaid
file is always created.
