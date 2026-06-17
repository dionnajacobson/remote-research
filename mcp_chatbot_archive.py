import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import nest_asyncio
from typing import TypedDict
from contextlib import AsyncExitStack

nest_asyncio.apply()
load_dotenv(override=True)

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:

    def __init__(self):
        self.sessions: list[ClientSession] = []
        # Tracks all async context managers (transports + sessions) so we can
        # close every MCP connection later with a single aclose() call.
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.available_tools: list[ToolDefinition] = []
        # Maps each tool name to the session that owns it (needed when multiple servers are connected).
        self.tool_to_session: dict[str, ClientSession] = {}

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    
    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)

            # enter_async_context is like `async with`, but the connection stays
            # open after this function returns. The stack remembers how to close
            # it later (in cleanup()) instead of closing at the end of a with-block.
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport

            # Same pattern for the MCP session: open now, close all sessions in
            # reverse order when exit_stack.aclose() runs.
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)
            
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]
        response = self.anthropic.messages.create(
            max_tokens=2024,
            model='claude-sonnet-4-6',
            tools=self.available_tools,
            messages=messages,
        )

        while response.stop_reason == 'tool_use':
            messages.append({'role': 'assistant', 'content': response.content})

            tool_results = []
            for block in response.content:
                if block.type == 'text':
                    print("Bot: ", block.text)
                elif block.type == 'tool_use':
                    print(f"Bot: Calling tool {block.name} with args {block.input}")
                    tool_args = block.input
                    tool_name = block.name
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    tool_result_text = result.content[0].text if result.content else "No result"
                    tool_results.append({
                        'type': 'tool_result',
                        'tool_use_id': block.id,
                        'content': tool_result_text,
                    })

            messages.append({'role': 'user', 'content': tool_results})
            response = self.anthropic.messages.create(
                max_tokens=2024,
                model='claude-sonnet-4-6',
                tools=self.available_tools,
                messages=messages,
            )

        for block in response.content:
            if block.type == 'text':
                print("Bot: ", block.text)

    
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Close every transport and session registered on the exit stack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() # new! 
        await chatbot.chat_loop()
    finally:
        # Always tear down MCP connections, even if chat_loop exits with an error.
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
