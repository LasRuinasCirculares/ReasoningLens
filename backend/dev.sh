export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080;http://192.168.14.73:5173;http://192.168.14.73:8080;http://192.168.14.73:3111"

# Enable reasoning analysis logging (set to 1 for debugging analysis results)
export REASONING_ANALYSIS_LOGS=1
export ENABLE_PERSISTENT_CONFIG="False"
export OPENAI_API_KEY="sk-LYWrR2EtydXV2gPcZAxEd71yEMD8MzCdonD8jpY757jt5cwu"
export OPENAI_API_BASE_URL="http://api.cipsup.cn/v1"
export DEFAULT_USER_ROLE="user"
export WEBUI_SECRET_KEY="sk-LYWrR2EtydXV2gPcZAxEd71yEMD8MzCdonD8jpY757jt5cwu"

PORT="${PORT:-8080}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
