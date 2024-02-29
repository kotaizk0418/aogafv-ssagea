import discord
import tempfile
from keras.models import load_model
from ml.sex import sex
import sys

# ボットのトークン
#TOKEN = open("token.txt", "r").read()
TOKEN = sys.argv[1]
# Discord Intentsの設定
intents = discord.Intents.default()
intents.message_content = True


# Discordクライアントを作成
client = discord.Client(intents=intents)

model_dir = "./data/cnn_sexDataV3ModelV12"
model     = load_model(model_dir)


@client.event
async def on_ready():
    print(f"Logged in as {client.user.name}")


@client.event
async def on_message(message):
    # メッセージがボット自身のものであれば無視
    if message.author == client.user:
        return

    
    # 画像が添付されているかチェック
    if message.text == "test":
        return await message.channel.send("ok")
    if len(message.attachments) > 0:
        if message.channel.id == 1212363487284830268:
            total_images = len(message.attachments)
            for index, attachment in enumerate(message.attachments, start=1):
                if attachment.content_type.startswith("image"):
                    # 画像のダウンロード
                    image_data = await attachment.read()
                    # 一時ファイルとして保存
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(image_data)
                    save_file_path = temp_file.name
                    result = sex(save_file_path, model=model)
                    text = f"メス: {result['メス']}\nオス: {result['オス']}"
                    print(message.channel.id)
                    return await message.channel.send(text)

                    


if __name__ == "__main__":
    # ボットを起動
    client.run(TOKEN)
