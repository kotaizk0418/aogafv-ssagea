import discord
import tempfile
from keras.models import load_model
from keep_alive import keep_alive
from ml.sex import sex
import subprocess
import json
import sys

# ボットのトークン
#TOKEN = open("token.txt", "r").read()
TOKEN = sys.argv[1]
# Discord Intentsの設定
intents = discord.Intents.all()


# Discordクライアントを作成
client = discord.Client(intents=intents)

model_dir = "./data/cnn_sexDataV3ModelV12"
model     = load_model(model_dir)


@client.event
async def on_ready():
    print(f"Logged in as {client.user.name}")
    if len(sys.argv) == 3:
        print("Restarted")
        target = client.get_channel(int(sys.argv[2]))
        print(target)
        try:
            await target.send("restarted.")
        except:
            pass


"""新規メンバー参加時に実行されるイベントハンドラ"""
@client.event
async def on_member_join(member):
    guild = member.guild
    channel = discord.utils.get(guild.text_channels, name="挨拶-greeting")

    return await channel.send(f'{member.mention} さんよろしくお願いします。\nルールチャンネルにて同意をお願いします。')


@client.event
async def on_message(message):
    # メッセージがボット自身のものであれば無視
    if message.author == client.user:
        return

    
    # 画像が添付されているかチェック
    if message.content == "test":
        return await message.channel.send("ok")
    
    elif message.content == "json":
        test_data = {
            "test": "text",
            "testn": 1
        }

        with open("test.json", "w") as f:
            json.dump(test_data, f)

        with open("test.json", "r") as f:
            r = json.load(f)
        print(r)
        return await message.channel.send("ok")
    
    elif message.content == "exit":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            await message.channel.send("exiting.")
            return sys.exit()

    elif message.content == "reboot":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            await message.channel.send("exiting.")
            subprocess.Popen(["python3", "discord_main.py", sys.argv[1], str(message.channel.id)])
            return sys.exit()

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
    keep_alive()
    client.run(TOKEN)
