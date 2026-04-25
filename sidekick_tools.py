try:
    from playwright.async_api import async_playwright
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from dotenv import load_dotenv
import os
import requests
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import markdown2
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False
from sqlalchemy import create_engine, text
import json


load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"


async def playwright_tools():
    if not PLAYWRIGHT_AVAILABLE:
        return [], None, None
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


def push(text: str):
    """Send a push notification to the user"""
    if not pushover_token or not pushover_user:
        return "Push notifications not configured (missing PUSHOVER_TOKEN/PUSHOVER_USER)"
    try:
        requests.post(pushover_url, data={"token": pushover_token, "user": pushover_user, "message": text})
        return "success"
    except Exception as e:
        return f"Push notification failed: {str(e)}"


def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


def markdown_to_pdf(markdown_filename: str, output_filename: str) -> str:
    """Converts Markdown file to PDF"""
    if not WEASYPRINT_AVAILABLE:
        return "PDF generation not available on this platform (WeasyPrint requires GTK on Windows)"
    if not os.path.exists(markdown_filename):
        return f"Error: File not found: {markdown_filename}"
    try:
        with open(markdown_filename, 'r', encoding='utf-8') as f:
            md_content = f.read()

        html_content = markdown2.markdown(md_content, extras=['fenced-code-blocks', 'tables', 'code-friendly'])
        styled_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; margin: 40px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>{html_content}</body>
</html>"""

        HTML(string=styled_html).write_pdf(output_filename)
        return f"Success! PDF created at: {output_filename}"
    except Exception as e:
        return f"Error during PDF conversion: {str(e)}"


def execute_sql_query(query: str) -> str:
    """Execute SQL query on SQLite database"""
    os.makedirs("sandbox", exist_ok=True)
    engine = create_engine("sqlite:///sandbox/data.db")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            conn.commit()
            if result.returns_rows:
                rows = [dict(row._mapping) for row in result]
                return json.dumps(rows, default=str)
            return "Query executed successfully"
    except Exception as e:
        return f"SQL error: {str(e)}"


async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()

    serper = GoogleSerperAPIWrapper()
    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    markdown_pdf_tool = Tool(
        name="markdown_to_pdf",
        func=markdown_to_pdf,
        description="""Converts an existing Markdown (.md) file to PDF format.
        Args:
        - markdown_filename (str): Path to the input .md file (e.g., 'sandbox/report.md')
        - output_filename (str): Path for the output .pdf file (e.g., 'sandbox/report.pdf')

        Note: The markdown file must already exist. Use write_file tool first to create the .md file,
        then use this tool to convert it to PDF. Both paths should typically be in the 'sandbox' directory."""
    )

    sql_tool = Tool(
        name="sql_query",
        func=execute_sql_query,
        description="""Execute SQL queries on SQLite database for data storage and retrieval.

        Use this to:
        - CREATE TABLE to store data
        - INSERT data into tables
        - SELECT to retrieve data
        - UPDATE or DELETE records

        Examples:
        - CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)
        - INSERT INTO users (name, email) VALUES ('John', 'john@example.com')
        - SELECT * FROM users WHERE name = 'John'
        - UPDATE users SET email = 'new@example.com' WHERE id = 1

        Database location: sandbox/data.db (auto-created)
        Returns: JSON for SELECT, success message for other queries"""
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
    python_repl = PythonREPLTool()

    return file_tools + [push_tool, tool_search, python_repl, wiki_tool, markdown_pdf_tool, sql_tool]
