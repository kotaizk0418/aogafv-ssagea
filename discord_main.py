import discord
import tempfile
from keras.models import load_model
from keep_alive import keep_alive
from ml.sex import sex
import subprocess
import psycopg2
import json
import sys
import os

# ボットのトークン
#TOKEN = open("token.txt", "r").read()
TOKEN = sys.argv[1]
# Discord Intentsの設定
intents = discord.Intents.all()


# Discordクライアントを作成
client = discord.Client(intents=intents)

model_dir = "./data/cnn_sexDataV3ModelV12"
model     = load_model(model_dir)

def get_connection():
    dsn = os.environ.get('DATABASE_URL')
    return psycopg2.connect(dsn=eval(dsn))

def calc_count(data):
    l = False
    data = list(data)
    #data[uid, level, tcount]
    data[2] += 1

    if data[2] % (data[1]*(30+data[1])) == 0:
        data[1] += 1
        data[2] = 0
        l = True
    return data, l

async def count_and_level_up_user(member):
    with get_connection() as conn:
        with conn.cursor() as cur:
            #cur.execute('SELECT * FROM userlevel;')
            #row = cur.fetchall()
            cur.execute('SELECT * FROM userlevel WHERE uid = %s', (str(member.author.id),))
            row = cur.fetchone()
            
            print(row)
            if row:
                d, l = calc_count(row)
                print(d)
                cur.execute("UPDATE userlevel SET (level, tcount) = (%s, %s) WHERE uid = %s", (d[1], d[2], d[0]))
                if l:
                    if d[1] > 9:
                        role = discord.utils.get(member.guild.roles, name="レベル10")
                        await member.author.add_roles(role)
                    return await member.channel.send("レベルアップ")
            elif not row:
                cur.execute("INSERT INTO userlevel VALUES (%s, %s, %s)", (str(member.author.id), 1, 1,))
                print("INSERT: %s".format(member.author.id))
                return
            
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM userlevel')
    return

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

    return await channel.send(f'{member.mention} さんよろしくお願いします。\n <#1212995044206968932> にて同意をお願いします。\n <#1212689053523378197> で自己紹介などしてくれると嬉しいです。')


@client.event
async def on_message(message):
    # メッセージがボット自身のものであれば無視
    if message.author == client.user:
        return

    if len(message.content) > 3:
        await count_and_level_up_user(message)

    # 画像が添付されているかチェック
    if message.content == "test":
        return await message.channel.send("ok")
    
    elif message.content == "user":
        return await message.channel.send(message.author.id)
    
    elif message.content == "embed":
        embed = discord.Embed(title="TITLE", description='', color=0xff0000)
        embed.add_field(name="", value="VALUE", inline=False)
        return await message.channel.send(embed=embed)
    
    elif message.content == "users":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" not in role_name_list:
            return
        
        text = ""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM userlevel;')
                row = cur.fetchall()
        
        for uid, lv, ct in row:
            pro = await client.fetch_user(uid)
            text += f"Name: {pro.name}" + "\n" \
                    f"Uid : {uid}"      + "\n" \
                    f"Level: {lv}"      + "\n" \
                    f"Count: {ct}"      + "\n\n"
        return await message.channel.send(text)
    
    elif message.content == "me":
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM userlevel WHERE uid = %s', (str(message.author.id),))
                row = cur.fetchone()
        row = list(row)
        return await message.channel.send(f"レベル　{row[1]} です。")

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
        #if True:
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
                    embed = discord.Embed(title="判定結果", description='', color=0x9e76b4)
                    embed.add_field(name="", value=text, inline=False)
                    return await message.channel.send(embed=embed)
    
                    


if __name__ == "__main__":
    # ボットを起動
    keep_alive()
    client.run(TOKEN)
