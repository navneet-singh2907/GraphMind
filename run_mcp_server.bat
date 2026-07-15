@echo off
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
if not exist "tmp" mkdir "tmp"
echo %date% %time% GraphMind MCP launcher started>>"tmp\graphmind-mcp-launch.log"
"venv\Scripts\python.exe" -m src.mcp_server
