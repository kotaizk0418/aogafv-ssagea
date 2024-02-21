import requests as req
from bs4 import BeautifulSoup as bs
import re
import os
import urllib.request
import urllib.error
import glob
from tqdm import tqdm

PIC_NUM = 3
DIR_NAME = '/Users/kota-izk/Documents/worksplace/Python/centipedeAI/male/'

def main():
    fetchurls = []
    filelist = glob.glob('*.png')
    [filelist.append(l) for l in glob.glob('*.jpg')]
    filelist.sort()
    print(filelist)
    print('ファイルアップロード中...')
    pbar_png = tqdm(total=len(filelist))
    for filename in filelist:
        fetchurls.append([upload(filename),filename])
        pbar_png.update(1)
    pbar_png.close()
    headers = { #UA偽装しないと検索結果がもらえません
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"
        }
    print('画像保存中...')
    for fetchurl in fetchurls:
        print(fetchurl[1])
        soup = get_content(fetchurl[0],headers)
        psurl = get_picsearchurl(soup,fetchurl[1])
        if not psurl==False:
            soup = get_content(psurl,headers)
            picurls = get_pic_url(soup)
            save_pic(picurls,fetchurl[1],headers)

def upload(filename):
    url = "https://www.google.co.jp/searchbyimage/upload" #googleの類似画像検索を使うにはここにアップロード
    multipart = {'encoded_image': (filename, open(filename, 'rb')), 'image_content': ''} #multipart/form-dataの形式らしい...
    response = req.post(url, files=multipart, allow_redirects=False)
    fetchurl = response.headers['Location']
    return fetchurl

def get_content(url,headers): #検索結果をgoogleさんから頂いてきます
    res = req.get(url,timeout=10000,headers=headers)
    soup = bs(res.content,'html.parser')
    return soup

def get_picsearchurl(soup,name): #画像のみの検索結果のURLを返します
    pic_search_urls = soup.find(class_='O1id0e')
    if re.findall('この画像の他のサイズは見つかりません。',str(pic_search_urls))==[]:
        pic_search_url = pic_search_urls.find('a').get('href')
        psurl = 'https://www.google.co.jp'+pic_search_url
        return psurl
    else:
        print(name,'の他のサイズは見つかりませんでした。')
        return False

def get_pic_url(soup): #類似画像検索の画像検索結果(画像が羅列される方の結果)の画像URL"リスト"を取得します。re.findallの(?:jpg|png)をpngに置換すればpng画像に絞れます
    tmp = str(soup)
    urls = re.findall(r'"https?.+\.(?:jpg|png).*",\d{3,},\d{3,}',tmp)
    return urls
    
def save_pic(urls,filename,headers): #先頭URLの画像を./DIR_NAMEに保存します。最初の被検索画像取得をrecursiveにするならこの保存URLは上階層とかに変えたほうがいいです
    for num in range(PIC_NUM):
        try:
            url = [s.strip('\'\" ') for s in urls[num].split(',')]
            if int(url[1])>576:
                tmp_name = filename.split('.')
                url_extension = os.path.splitext(url[0])
                ext = re.sub(r'\.','',str(url_extension[1]))
                ext = re.sub(r'\W.*','',ext)
                path = DIR_NAME+'/'+tmp_name[0]+'_'+'{0:02d}'.format(num)+'.'+ext
                dir_exist = os.path.isdir('./'+DIR_NAME)
                if not dir_exist:
                    os.mkdir('./'+DIR_NAME)
                file_exist = os.path.isfile(path)
                if not file_exist:
                    try:
                        request = urllib.request.Request(url[0], headers=headers)
                        with urllib.request.urlopen(request) as web_file, open(path, 'wb') as local_file:
                            local_file.write(web_file.read())
                    except urllib.error.URLError as e:
                        print(e)
            else:
                print('低画質画像しか見つからなかったため',filename,'を保存しませんでした\n')
        except:
            pass

if __name__ == '__main__':
    main()