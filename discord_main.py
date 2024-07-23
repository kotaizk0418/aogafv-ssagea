import discord
import tempfile
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keep_alive import keep_alive
from ml.sex import sex
import subprocess
from datetime import datetime, timedelta, timezone
from dateutil import tz
from zoneinfo import ZoneInfo
import asyncio
import psycopg2
import json
import emoji
import sys
import logging
import time
import os

import tensorflow as tf
from collections import Counter
import torch
import cv2

from image_text_processor import Title, Text, Inline, image_processing


class CustomImageDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []
        for i in index_array:
            image_path = self.filepaths[i]
            img = load_valid_image(image_path)
            if img is not None:
                x = self.image_data_generator.standardize(self.image_data_generator.random_transform(img))
                batch_x.append(x)
                batch_y.append(self.labels[i])
        return np.array(batch_x), np.array(batch_y)

async def train_model(existing_model, train_generator, val_generator, batch_size, epoch):
    loop = asyncio.get_event_loop()
    tf.config.run_functions_eagerly(True)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    train_labels = train_generator.classes
    counter = Counter(train_labels)
    max_count = float(max(counter.values()))
    class_weight = {class_id: max_count/num_images for class_id, num_images in counter.items()}
    history = await loop.run_in_executor(
        None,
        lambda: existing_model.fit(
            train_generator,
            validation_data=val_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_steps=val_generator.samples // batch_size,
            epochs=epoch,
            class_weight=class_weight,
            callbacks=callbacks
        )
    )
    return history

DATA_PATH = os.environ.get("DATA_PATH") if os.environ.get("DATA_PATH") else ""
SELECTED_MODEL = ""
# ボットのトークン
#TOKEN = open("token.txt", "r").read()
TOKEN = sys.argv[1]
# Discord Intentsの設定
intents = discord.Intents.all()


# Discordクライアントを作成
client = discord.Client(intents=intents)

model_dir = "./models/path_to_save_updated_model.h5"
model_dir = "./data/model_b.h5"
model     = load_model(model_dir)

def get_connection():
    dsn = os.environ.get('DATABASE_URL')
    dsn = "'postgresql://discord_bot_data_user:BdimL061db6iEusp0P8ftr4OnyyLj2Th@dpg-cnoph1vsc6pc73b7d3ug-a.oregon-postgres.render.com/discord_bot_data'"
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


with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM model WHERE id = 1;")
            row = cur.fetchone()
            SELECTED_MODEL = row[1]

print(SELECTED_MODEL)

def update_select_model(model_path):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE model SET model = (%s) WHERE id = 1", (model_path,))
            cur.execute("SELECT * FROM model WHERE id = 1;")
            row = cur.fetchone()
            SELECTED_MODEL = row[1]
            print("update to", SELECTED_MODEL)
            
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
                    await send_logs(f"{member.author.id} の経験値が5上がりました。\n{d}: {l}")
                dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
                if dt2.hour in [19, 20, 21, 22]:
                    await send_logs(f"{member.author.id} の経験値が2上がりました。\n{d}: {l}")
                print(d)
                cur.execute("UPDATE userlevel SET (level, tcount) = (%s, %s) WHERE uid = %s", (d[1], d[2], d[0]))
                logging.info("UPDATE userlevel SET (level, tcount) = (%s, %s) WHERE uid = %s".format(d[1], d[2], d[0]))
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
                logging.info("INSERT: %s".format(member.author.id))
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

def count_files(directory):
    # 指定されたディレクトリ内のファイルリストを取得
    try:
        files = os.listdir(directory)
        # ファイルの数をカウント
        file_count = len(files)
        return file_count
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return -1
# @client.event
# async def on_voice_state_update(member, before, after):
#     print(after.channel)
#     if before.channel is None and after.channel is not None:
#         target = client.get_channel("1229365047206219887")
#         await target.send(f"<@&1227926580869988404> VCが開かれました。\n#{after.channel}")

"""新規メンバー参加時に実行されるイベントハンドラ"""
@client.event
async def on_member_join(member):
    
    guild = member.guild
    channel = discord.utils.get(guild.text_channels, name="挨拶-greeting")
    
    # インバイトをループして、参加したメンバーがどのインバイトを使用したかを確認
    print(f"{member.guild.id}")
            
    return await channel.send(f'{member.mention} さんよろしくお願いします。\n <#1212995044206968932> にて同意をお願いします。\n <#1212689053523378197> で自己紹介などしてくれると嬉しいです。')

@client.event
async def on_thread_create(channel):
    
    print(f'create forum: {channel.name} ({channel.owner_id})')
    print(dir(channel))
    #await channel.send(f"{channel.guild.roles}")
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
                    await send_logs(f"{channel.owner_id} の経験値が5上がりました。\n{d}: {l}")
                dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
                if dt2.hour in [19, 20, 21, 22]:
                    await send_logs(f"{channel.owner_id} の経験値が2上がりました。\n{d}: {l}")
                print(d)
                cur.execute("UPDATE userlevel SET (level, tcount) = (%s, %s) WHERE uid = %s", (d[1], d[2], d[0]))
                if l:
                    if d[1] > 9:
                        role = discord.utils.get(channel.guild.roles, name="レベル10")
                        await channel.owner.add_roles(role)
                    elif d[1] > 2:
                        role = discord.utils.get(channel.guild.roles, name="レベル3")
                        await channel.owner.add_roles(role)
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
    global SELECTED_MODEL
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
    
    elif message.content == "my role":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        print(role_name_list)
        return await message.channel.send(f"{role_name_list}")
    
    elif message.content == "accept":
        role = discord.utils.get(message.channel.guild.roles, name="メンションされていい人")
        await message.author.add_roles(role)
        return await message.channel.send("ok.")
    
    elif message.content == "hunter":
        role = discord.utils.get(message.channel.guild.roles, name="採り子")
        await message.author.add_roles(role)
        return await message.channel.send("ok.")

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
        
    elif message.content == "dataset":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            male_count = count_files(DATA_PATH+"dataset/male/")
            female_count = count_files(DATA_PATH+"dataset/female/")
            return await message.channel.send(f"male: {male_count}\nfemale: {female_count}")
    
    elif message.content == "models":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            files = os.listdir(DATA_PATH+"models")
            text = ""
            for file in files:
                text += f"- {file}\n"
            return await message.channel.send(text)
        
    elif message.content.startswith("model "):
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            text = message.content.split()
            path = DATA_PATH + f"models/{text[1]}"
            print(path)
            try:
                load_model(path)
            except:
                return await message.channel.send("error")
            SELECTED_MODEL = path
            update_select_model(path)
            return await message.channel.send("ok")
    
    elif message.content == "eval":
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            model = load_model(SELECTED_MODEL)
            
            datagen = CustomImageDataGenerator(
                rescale=1./255,  # 画像のリスケーリング
                shear_range=0.2,  # シアリングの範囲
                zoom_range=0.2,  # ズームの範囲
                horizontal_flip=True  # 水平方向の反転
            )
            male_count   = count_files(DATA_PATH+"dataset/male/")
            female_count = count_files(DATA_PATH+"dataset/female/")
            batch_size = 2
            dataset_path = DATA_PATH+"dataset/"

            test_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='binary'
            )
            loss, accuracy = model.evaluate(test_generator, verbose=0)

            return await message.channel.send(f"model: {SELECTED_MODEL}\nacc: {accuracy}\nloss: {loss}")

    elif message.content.startswith("train "):
        chkrls = message.author.roles
        role_name_list = []
        for role in chkrls:  # roleにはRoleオブジェクトが入っている
            role_name_list.append(role.name)
        
        if "管理者" in role_name_list:
            text = message.content
            epoch = int(text.split()[1])
            batch_size = int(text.split()[2])
            existing_model = load_model('data/model_b.h5')
            dataset_path = DATA_PATH+"dataset/"

            await message.channel.send("train start..")

            datagen = CustomImageDataGenerator(
                rescale=1./255,  # 画像のリスケーリング
                shear_range=0.2,  # シアリングの範囲
                zoom_range=0.2,  # ズームの範囲
                horizontal_flip=True,  # 水平方向の反転
                validation_split=0.5 
            )
            male_count   = count_files(DATA_PATH+"dataset/male/")
            female_count = count_files(DATA_PATH+"dataset/female/")
            batch_size = batch_size

            train_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='binary'
            )
            val_generator = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=batch_size,
                class_mode='binary'
            )
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.0001,
                decay_steps=300,
                decay_rate=0.6
            )
            existing_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                       loss='binary_crossentropy',
                       metrics=['accuracy'],
                       run_eagerly=True)

            history = await train_model(existing_model, train_generator, val_generator, batch_size, epoch)

            train_acc = history.history['accuracy'][-1]
            train_loss = history.history['loss'][-1]
            await message.channel.send(f"train acc: {train_acc}\nloss: {train_loss}")
            dt2 = datetime.now(ZoneInfo("Asia/Tokyo"))
            existing_model.save(DATA_PATH+f'models/{dt2.month}-{dt2.day}_updated_model.h5')
            await message.channel.send(file=discord.File(DATA_PATH+f'models/{dt2.month}-{dt2.day}_updated_model.h5'))

    if len(message.attachments) > 0:
        if message.channel.id in [1259256195823308872, 1259187729607037040]: #female train
            for index, attachment in enumerate(message.attachments, start=1):
                if attachment.content_type.startswith("image"):
                    # 画像のダウンロード
                    image_data = await attachment.read()
                    # 一時ファイルとして保存
                    with open(DATA_PATH+f"dataset/female/{attachment.filename}.jpg", "wb") as f:
                        f.write(image_data)
            return await message.channel.send("提供ありがとうございます。雌雄を間違えた場合は送信した画像は絶対に削除せず、画像に対してリプライで「ミス」と送信しておいてください。")
            
        if message.channel.id in [1259256210792775853, 1259187630478856292]: #male train
            for index, attachment in enumerate(message.attachments, start=1):
                if attachment.content_type.startswith("image"):
                    # 画像のダウンロード
                    image_data = await attachment.read()
                    # 一時ファイルとして保存
                    with open(DATA_PATH+f"dataset/male/{attachment.filename}.jpg", "wb") as f:
                        f.write(image_data)
            return await message.channel.send("提供ありがとうございます。雌雄を間違えた場合は送信した画像は絶対に削除せず、画像に対してリプライで「ミス」と送信しておいてください。")
        
        if message.channel.id in [1212363487284830268, 1258773854478798880, 1258776074326769664, 1258786734846775307, 1213395104551669760]:
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
                    if not centioede_check(save_file_path):
                        return await message.channel.send("ムカデを検出できませんでした")
                    await message.channel.send("判定中..")
                    result = sex(save_file_path, model=load_model(SELECTED_MODEL), augment_times=10)
                    print(message.channel.id)

                    objects = [
                        Title("判定結果", bold=True),
                        Title(" ", bold=True)
                    ]

                    for k,v in result.items():
                        objects.append(Inline([Text(f"{k} :  ", bold=True), Text(v, bold=True)]))
                    
                    image_processing(save_file_path, objects, vertical_alignment="middle")
                    msg = await message.channel.send(file=discord.File("output.jpg"))
                    # embed = discord.Embed(title="判定結果", description="100%正しいわけではありません。avgとは画像を拡張し、推論した結果の平均値です。", color=0x9e76b4)
                    # embed.add_field(name="", value="", inline=False)
                    # for k, v in result.items():
                    #     embed.add_field(name=k, value=v, inline=False)
                    # embed.add_field(name="", value="", inline=False)
                    # embed.add_field(name="", value="正しければ⭕️、外れていれば❌、不明な場合は❓", inline=False)
                    # msg = await message.channel.send(embed=embed)
                    msg = await message.channel.send("100%正しいわけではありません。avgとは画像を拡張し、推論した結果の平均値です。\n正しければ⭕️、外れていれば❌、不明な場合は❓をお願いします。")
                    for k, v in {'⭕': 'a', '❌': '️b', '❓': 'c'}.items():
                        await msg.add_reaction(k)
    
                    
def centioede_check(image_path):
    # モデルの読み込み
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='ml/best_model.pt')
    
    # 画像の読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 推論
    results = model(img_rgb)
    
    # 結果の取得
    results_list = results.xyxy[0].cpu().numpy()  # [[xmin, ymin, xmax, ymax, confidence, class], ...]
    print(results_list)
    
    if len(results_list) == 0:
        print("No objects detected.")
        return False
    return True

if __name__ == "__main__":
    # ボットを起動
    keep_alive()
    client.run(TOKEN)
