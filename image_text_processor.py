from PIL import Image, ImageDraw, ImageFont, ImageFilter

class Title:
    def __init__(self, text, bold=False):
        self.text = text
        self.bold = bold
        self.is_title = True

class Text:
    def __init__(self, text, bold=False):
        self.text = text
        self.bold = bold
        self.is_title = False

class Inline:
    def __init__(self, objects):
        self.objects = objects


def calculate_font_size(img_width, img_height, scale_factor=0.05):
    # 画像の幅と高さに基づいてフォントサイズを計算
    aspect = img_width / img_height
    size = int((img_width + img_height) / 2 * scale_factor)
    
    # アスペクト比に応じてフォントサイズを調整
    if aspect > 0.6:
        size = int(size * 1.2)  # アスペクト比が高い場合にフォントサイズを拡大
    else:
        size = int(size * 0.8)  # アスペクト比が低い場合にフォントサイズを縮小
    
    return size

def image_processing(img_path, objects, arrangement="center", vertical_alignment="top", font_size=None):
    """
    指定された画像にテキストを書き込む関数。
    
    Parameters:
        img_path (str): 画像のパス
        objects (list): Title, TextまたはInlineオブジェクトのリスト
        arrangement (str): テキストの配置方法 ("center", "left", "right" をサポート)
        font_size (int): テキストのフォントサイズ（デフォルトは自動的に計算）
    """
    # 画像を開く
    img = Image.open(img_path)
    # 画像全体にブラーをかける
    img = img.filter(ImageFilter.GaussianBlur(20))
    draw = ImageDraw.Draw(img)

    # 画像サイズに基づいてフォントサイズを決定
    img_width, img_height = img.size
    if font_size is None:
        aspect = img_width / img_height
        size = 15 if aspect > 0.6 else 25
        base_font_size = calculate_font_size(img_width, img_height, scale_factor=0.067)  # 自動的にフォントサイズを調整
    else:
        base_font_size = font_size

    # デフォルトフォントの設定
    def get_font(size, bold=False):
        try:
            font_path = "fonts/arialbd.ttf" if not bold else "fonts/arial.otf"
            return ImageFont.truetype(font_path, size)
        except IOError:
            return ImageFont.load_default()

    title_font = get_font(base_font_size * 2, bold=True)
    bold_font = get_font(base_font_size, bold=True)
    regular_font = get_font(base_font_size)

    def draw_text(draw, x, y, text, font, vertical_alignment='top'):
        text_width, text_height = draw.textsize(text, font=font)

        if vertical_alignment == 'middle':
            y -= text_height // 2
        elif vertical_alignment == 'bottom':
            y -= text_height

        draw.text((x, y), text, font=font, fill="white", stroke_width=2, stroke_fill="black")
        return text_width, text_height

    # テキストの配置のための高さを計算
    y = 50  # 初期のy座標を調整

    for obj in objects:
        if isinstance(obj, Inline):
            texts = obj.objects
            total_width = 0
            text_sizes = []
            for text_obj in texts:
                if isinstance(text_obj, Title):
                    font = title_font
                    text = text_obj.text
                elif isinstance(text_obj, Text):
                    font = bold_font if text_obj.bold else regular_font
                    text = text_obj.text
                else:
                    continue

                text_width, text_height = draw.textsize(text, font=font)
                text_sizes.append((text, font, text_width, text_height))
                total_width += text_width

            if arrangement == "center":
                x = (img_width - total_width) // 2
            elif arrangement == "right":
                x = img_width - total_width - 10
            else:  # "left" またはデフォルト
                x = 10

            for text, font, text_width, text_height in text_sizes:
                if x + text_width > img_width:
                    x = 10
                    y += text_height + 10
                draw_text(draw, x, y, text, font, vertical_alignment=vertical_alignment)  # ここで高さを指定
                x += text_width

            y += text_sizes[0][3] + 20  # 次の行のための余白を追加

        else:
            if isinstance(obj, Title):
                font = title_font
                text = obj.text
            elif isinstance(obj, Text):
                font = bold_font if obj.bold else regular_font
                text = obj.text
            else:
                continue

            text_width, text_height = draw.textsize(text, font=font)
            if arrangement == "center":
                x = (img_width - text_width) // 2
            elif arrangement == "right":
                x = img_width - text_width - 10
            else:  # "left" またはデフォルト
                x = 10

            if x + text_width > img_width:
                x = 10
                y += text_height + 10
            draw_text(draw, x, y, text, font, vertical_alignment='top')  # ここで高さを指定
            y += text_height + 20  # 次の行のための余白を追加

    img.save("output.jpg")

if __name__ == "__main__":
    objects = [
        Title("Title Text"), 
        Inline([Text("Bold Text", bold=True), Text("Regular Text", bold=False)]),
        Text("Another Line", bold=False)
    ]

    image_processing("DSC_0492.jpg", objects, arrangement="center", vertical_alignment="middle")
