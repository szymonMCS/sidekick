---
title: Sidekick
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---
# 🤖 Sidekick Personal Co-Worker

An intelligent AI assistant with multiple specialized agents that helps with daily tasks through web browsing, code execution, file management, and communication.

## ✨ Features

### 🔐 Multi-User System
- User registration and login
- Secure password hashing (bcrypt)
- Separate conversation history for each user
- User session management

### 🧠 Specialized Agents
Sidekick uses an intelligent routing system that automatically delegates tasks to the appropriate specialists:

- **Research Agent** - web searching, Wikipedia, web browsing
- **Coding Agent** - Python code execution, file operations, document creation
- **Communication Agent** - sending push notifications, communication management
- **General Agent** - general queries and conversations

### 🛠️ Available Tools

#### Web Browsing
- Automatic web page navigation (Playwright)
- Google search (Serper API)
- Wikipedia API

#### File Operations
- Creating, reading, copying, moving, and deleting files
- Automatic Markdown → PDF conversion
- Directory management

#### Code Execution
- Python REPL - real-time Python code execution
- Full access to Python libraries

#### Database
- SQLite for storing user data
- SQL query tool for custom queries
- Conversation memory (LangGraph checkpointer)

#### Communication
- Push notifications (Pushover)

### 💡 Intelligent Features

- **Clarifying Questions** - agent automatically asks up to 3 clarifying questions before starting a task
- **Evaluator** - checks whether the response meets success criteria
- **Feedback Loop** - agent improves its work based on feedback if criteria are not met
- **Specialist Routing** - coordinator directs tasks to the best-matched agent

## 📋 Requirements

### Python Dependencies
```
gradio
langgraph
langchain
langchain-openai
langchain-community
langchain-experimental
playwright
python-dotenv
bcrypt
markdown2
pdfkit
sqlalchemy
requests
```

### Additional Requirements
- **wkhtmltopdf** - for PDF conversion (installed in `C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe`)
- **Playwright browsers** - install via `playwright install chromium`

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root directory:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Pushover (for push notifications)
PUSHOVER_TOKEN=your_pushover_token
PUSHOVER_USER=your_pushover_user_key

# Serper API (for Google search)
SERPER_API_KEY=your_serper_api_key
```

### 2. Installing Dependencies

```bash
# Install Python packages
pip install gradio langgraph langchain langchain-openai langchain-community langchain-experimental playwright python-dotenv bcrypt markdown2 pdfkit sqlalchemy requests

# Install Playwright browsers
playwright install chromium

# Install wkhtmltopdf
# Windows: Download from https://wkhtmltopdf.org/downloads.html
# Linux: sudo apt-get install wkhtmltopdf
# macOS: brew install wkhtmltopdf
```

## 🚀 Running

```bash
python app.py
```

The application will launch in the browser with a Gradio interface.

## 📁 Project Structure

```
sidekick/
├── app.py                  # Main Gradio application with UI
├── sidekick.py            # Main agent logic and LangGraph graph
├── sidekick_tools.py      # Tool definitions (Playwright, files, SQL, etc.)
├── user_auth.py           # User authentication system
├── sandbox/               # Working directory for files
│   ├── users.db          # User database
│   ├── memory_*.db       # Conversation memory for each user
│   └── data.db           # SQL database for sql_query tool
└── .env                   # Environment variables (not committed)
```

## 🎯 How to Use

1. **Run the application**: `python app.py`
2. **Register** or **log in** to your account
3. **Enter a task** in the text field
4. **Specify success criteria** (optional) - what you consider success
5. **Click "Go!"** or press Enter
6. **Answer clarifying questions** (if the agent asks)
7. **Watch the agent work** - you'll see evaluator feedback and final results

### Example Tasks

```
"Find the latest information about GPT-4 and save it to a file"
Criteria: Markdown file with summary and PDF conversion

"Write Python code that calculates the factorial of 10"
Criteria: Working code with result

"Search for a carbonara recipe and send me a notification"
Criteria: Notification with recipe

"Create a report of the 5 most important tech news today"
Criteria: PDF document with summary
```

## 🏗️ Architecture

The project uses **LangGraph** to orchestrate agents in a flow graph:

```
START → Clarification Agent → Coordinator
         ↓ (if questions)          ↓
        END                   Specialist Router
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Research Agent   Coding Agent   Communication Agent
                    └───────────────┼───────────────┘
                                    ↓
                              Tool Execution
                                    ↓
                              Evaluator
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
              Back to Coordinator              SUCCESS → END
```

### Key Components

- **State Management** - TypedDict with LangGraph for tracking conversation state
- **SqliteSaver** - checkpointer for saving conversation history
- **Structured Output** - Pydantic models for guaranteed response structure
- **Tool Binding** - dynamic tool binding with LLM

## 🔧 Customization

### Adding New Tools

In the [sidekick_tools.py](sidekick_tools.py) file:

```python
from langchain_core.tools import Tool

def my_custom_function(param: str) -> str:
    # Your logic
    return "result"

my_tool = Tool(
    name="my_tool_name",
    func=my_custom_function,
    description="Tool description for LLM"
)

# Add to tool list in other_tools() function
```

### Adding a New Specialist

1. Create a new agent function in [sidekick.py](sidekick.py)
2. Add a node to the graph in `build_graph()`
3. Update routing in `coordinator_agent()`

## 📝 Notes

- **Memory** - each user has a separate conversation memory database
- **Sandbox** - all file operations are performed in the `sandbox/` directory
- **Automatic PDF** - every Markdown file is automatically converted to PDF
- **Browser** - Playwright runs the browser in headless=False mode (visible)

## 🐛 Troubleshooting

### Missing `execute_sql_query` function
The function is used in [sidekick_tools.py:119](sidekick_tools.py#L119) but is not defined. It should be added:

```python
def execute_sql_query(query: str) -> str:
    """Execute SQL query on sandbox/data.db"""
    try:
        engine = create_engine('sqlite:///sandbox/data.db')
        with engine.connect() as conn:
            result = conn.execute(text(query))
            if query.strip().upper().startswith('SELECT'):
                return json.dumps([dict(row) for row in result.mappings()])
            conn.commit()
            return "Query executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"
```

### wkhtmltopdf issues
Make sure the path in [sidekick_tools.py:82](sidekick_tools.py#L82) is correct for your system.

## 📄 License

Private project.

## 🤝 Contact

Szymon - LLM project

---

**Powered by**: OpenAI GPT-4o-mini, LangChain, LangGraph, Gradio
