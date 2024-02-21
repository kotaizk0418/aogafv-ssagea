from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageMessage, FollowEvent, FlexSendMessage
)
from keras.models import load_model
from pathlib import Path
import json
import os

from api import Commands
from ml.sex import sex

app = Flask(__name__)


#環境変数取得
YOUR_CHANNEL_ACCESS_TOKEN = "IWuOplto6zwnBOoWXjr3EDJRl4s61KqJoNXt5/QYDfTKpT5/UdXhb4c3PzAQqz8kglZRMqKSwlg7hFrVHy4Hn25CM9NqVP2DL3iWuyaktELfN52knujfXnplvA/8KIM0Lc6TXxSlBph3w5o7+wjQgAdB04t89/1O/w1cDnyilFU="
YOUR_CHANNEL_SECRET = "ebcf9dbb6fced81422891a6d2fe471a6"

line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)
command = Commands(line_bot_api)

model_dir = "data/cnn_sexDataV3ModelV6"
model     = load_model(model_dir)

@app.route("/")
def index():
    with open("preview-contents/index.html") as f:
        return f.read()


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    res = command._parse_event(event)

    if res == "ok":
        return
    
with open("preview-contents/help_flex.json", "r") as f:
    help_flex = json.load(f)


@handler.add(FollowEvent)
def handle_message(event):
    line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="挨拶", contents=help_flex))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print(event)
    message_id = event.message.id

    # message_idから画像のバイナリデータを取得
    message_content = line_bot_api.get_message_content(message_id)

    with open(Path(f"static/images/{message_id}.jpg").absolute(), "wb") as f:
        # バイナリを1024バイトずつ書き込む
        for chunk in message_content.iter_content():
            f.write(chunk)
    
    l = sex(f"static/images/{message_id}.jpg", model=model)

    with open("preview-contents/result_flex.json", "r") as f:
        content = json.load(f)

    content["contents"][0]["body"]["contents"][4]["contents"][1]["text"] = l["メス"]
    content["contents"][0]["body"]["contents"][5]["contents"][1]["text"] = l["オス"]
    return line_bot_api.reply_message(
            event.reply_token,
            FlexSendMessage(alt_text="判定結果", contents=content))


if __name__ == "__main__":
#    app.run()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)