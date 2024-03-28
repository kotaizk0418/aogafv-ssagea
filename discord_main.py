import discord
import tempfile
from keras.models import load_model
from keep_alive import keep_alive
from ml.sex import sex
import subprocess
from datetime import datetime, timedelta, timezone
from dateutil import tz
from zoneinfo import ZoneInfo
import psycopg2
import json
import emoji
import sys
import logging
import time
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
    #return psycopg2.connect(dsn=dsn)

async def send_logs(text):
    id = 1222867960637554819
    target = client.get_channel(id)
    dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
    text = f"-{dt2}\n" + \
           f"`{text}`"
    return await target.send(text)

def calc_count(data, x5):
    l = False
    data = list(data)
    #data[uid, level, tcount]
    if x5:
        data[2] += 5
    else:
        dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
        if dt2.hour in [19, 20, 21, 22]:
            data[2] += 2
        else:
            data[2] += 1

    if data[2] >= data[1]*(30+data[1]):
        data[1] += 1
        data[2] = 0
        l = True
    return data, l

async def count_and_level_up_user(member, x5):
    with get_connection() as conn:
        with conn.cursor() as cur:
            #cur.execute('SELECT * FROM userlevel;')
            #row = cur.fetchall()
            cur.execute('SELECT * FROM userlevel WHERE uid = %s', (str(member.author.id),))
            row = cur.fetchone()
            
            print(row)
            if row:
                d, l = calc_count(row, x5)
                if x5:
                    await send_logs(f"{member.author.id} の経験値が5上がりました。")
                dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
                if dt2.hour in [19, 20, 21, 22]:
                    await send_logs(f"{member.author.id} の経験値が2上がりました。")
                print(d)
                cur.execute("UPDATE userlevel SET (level, tcount) = (%s, %s) WHERE uid = %s", (d[1], d[2], d[0]))
                if l:
                    if d[1] > 9:
                        role = discord.utils.get(member.guild.roles, name="レベル10")
                        await member.author.add_roles(role)
                    elif d[1] > 2:
                        role = discord.utils.get(member.guild.roles, name="レベル3")
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
async def on_thread_create(channel):
    
    print(f'create forum: {channel.name} ({channel.owner_id})')
    print(dir(channel))
    x5 = True
    with get_connection() as conn:
        with conn.cursor() as cur:
            #cur.execute('SELECT * FROM userlevel;')
            #row = cur.fetchall()
            cur.execute('SELECT * FROM userlevel WHERE uid = %s', (str(channel.owner_id),))
            row = cur.fetchone()
            
            print(row)
            if row:
                d, l = calc_count(row, x5)
                if x5:
                    await send_logs(f"{channel.owner_id} の経験値が5上がりました。")
                dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
                if dt2.hour in [19, 20, 21, 22]:
                    await send_logs(f"{channel.owner_id} の経験値が2上がりました。")
                print(d)
                cur.execute("UPDATE userlevel SET (level, tcount) = (%s, %s) WHERE uid = %s", (d[1], d[2], d[0]))
                if l:
                    if d[1] > 9:
                        role = discord.utils.get(channel.guild.roles, name="レベル10")
                        await channel.author.add_roles(role)
                    elif d[1] > 2:
                        role = discord.utils.get(channel.guild.roles, name="レベル3")
                        await channel.author.add_roles(role)
                    return await channel.send("レベルアップ")
            elif not row:
                cur.execute("INSERT INTO userlevel VALUES (%s, %s, %s)", (str(channel.owner_id), 1, 1,))
                print("INSERT: %s".format(channel.owner_id))
                return
            
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM userlevel')
    return
    


@client.event
async def on_message(message):
    # メッセージがボット自身のものであれば無視
    if message.author == client.user:
        return

    if len(message.content) > 3:
        if message.channel.id == 1221702474344169562:
            x5 = True
        else:
            x5 = False
        
        try:
            await count_and_level_up_user(message, x5)
        except:
            pass
    # 画像が添付されているかチェック
    if message.content == "test":
        ctime = time.ctime()
        dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
        return await message.channel.send(f"{dt2.hour}: {ctime}")
    
    elif message.content == "user":
        return await message.channel.send(message.author.id)
    

    elif message.content.startswith("poll "):

        title = message.content.split()[1]
        value = "".join(message.content.split()[2:])
        emojis = [c for c in value if c in emoji.EMOJI_DATA]
        # 絵文字ごとにメッセージを分割
        splitted_messages = [value]
        for emoji_char in emojis:
            new_splitted_messages = []
            for part in splitted_messages:
                new_splitted_messages.extend(part.split(emoji_char))
            splitted_messages = new_splitted_messages
        
        # 絵文字と対応する文字列の辞書を作成
        emoji_dict = {}
        for emoji_char, part in zip(emojis, splitted_messages):
            emoji_dict[emoji_char] = part
        
        # 辞書を表示
        print(emoji_dict)
        embed = discord.Embed(title=title, description="", color=0xfff000)
        pro = await client.fetch_user(message.author.id)
        
        embed.set_author(name=pro.display_name, # Botのユーザー名
                     url="", # titleのurlのようにnameをリンクにできる。botのWebサイトとかGithubとか
                     icon_url=message.author.avatar # Botのアイコンを設定してみる
                     )
        for k, v in emoji_dict.items():
            embed.add_field(name=f"{k} {v}", value="", inline=False)
        
        msg = await message.channel.send(embeds=[embed])
        
        for k, v in emoji_dict.items():
            await msg.add_reaction(k)
    
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
