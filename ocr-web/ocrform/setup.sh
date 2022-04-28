mkdir -p ~/.streamlit/

echo "
[theme]
base='light'
primaryColor='#146b48'
secondaryBackgroundColor='#e7f7f7'
[server]
port = $PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
