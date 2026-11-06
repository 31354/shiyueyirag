from fastmcp import FastMCP

mcp = FastMCP("External Demo Server")

@mcp.tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city} 今天晴，25°C"

if __name__ == "__main__":
    mcp.run(transport="stdio")