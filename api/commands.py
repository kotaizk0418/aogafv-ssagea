from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage
)
import json

with open("preview-contents/help_flex.json", "r") as f:
    help_flex = json.load(f)

class Commands:
    def __init__(self, line):
        self.line_bot_api = line
        self.commands     = self._parse()
    
    def _parse(self):
        all_function = Commands.__dict__

        result = {}
        for k, v in all_function.items():
            if k.startswith("_"):
                pass
            result[v.__doc__] = v
        return result

    def _parse_event(self, event):
        text = event.message.text
        if text in self.commands:
            return self.commands[text](self, event)

    def test(self, event):
        """test"""
        self.line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="ok"))
        return "ok"
    
    def help(self, event):
        """help"""
        self.line_bot_api.reply_message(event.reply_token, FlexSendMessage(alt_text="挨拶", contents=help_flex))
        return "ok"
