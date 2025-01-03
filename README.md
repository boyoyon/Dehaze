<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>Dehaze</center></h1>
        <h2>なにものか？</h2>
        <p>
            モヤの掛かった画像からモヤを除去した画像を生成します。<br />
            (実行結果１) python src\dehaze.py data\hazy3.png<br />
            <img src="images/dehaze1.png"><br>
            露出不足画像の補正にも使えます<br />
            (実行結果２)python src\inv-dehaze-inv.py data\hazy6.png<br />
            <img src="images/dehaze2.png"><br>
        </p>
        <h3>処理概要</h3>
        <p>
            ResNet などで有名な Kaiming He さんの Dark Channel Prior の簡易実装です。<br>
            <a href="https://people.csail.mit.edu/kaiming/publications/cvpr09.pdf">Single Image Haze Removal Using Dark Channel Prior</a><br>
            同じく Kaiming He さんのGuided Image Filtering を Dark Channel の精緻化に使っています。<br>
            <a href="https://img.shlab.org.cn/pjlab/files/2023/12/638387759189530000.pdf">Guided Image Filtering</a><br>
            <img src="images/dehaze2.svg"><br>
        </p>
        <p>
            露出不足の画像を明暗反転するとモヤの掛かった画像のように見える、<br>
            という思い付き的な発想で、露出不足改善に dehaze を利用することもできます。<br>
            <a href="https://www.researchgate.net/profile/Shengping-Zhang-5/publication/271554887_Night_video_enhancement_using_improved_dark_channel_prior">NIGHT VIDEO ENHANCEMENT USING IMPROVED DARK CHANNEL PRIOR (2013)</a><br>
            <img src="images/dehaze3.svg"><br>
        </p>
        <h2>環境構築方法</h2>
        <p>
            pip install opencv-python opencv-contrib-python<br>
            <br>
            ImportError: cannot import name 'guidedFilter' from 'cv2.ximgproc' (unknown location)<br>
            となる場合は<br>
            pip uninstall opencv-python opencv-contrib-python<br>
            pip install opencv-python opencv-contrib-python<br>
            としてみてください。<br>
            <br>
            動作速度が遅いので、高速化したい場合は src/cpp のソースをビルドしてください。
        </p>
        <h2>使い方</h2>
        <h3>モヤの除去</h3>
        <p>
            python dehaze.py (画像ファイル名) [(weight：1～99：デフォルト70) (window_size：デフォルト45)]<br>
            <br>
            入力画像、dehazeされた画像が表示されます。<br>
            ESCキー押下でプログラム終了。dehazed_(画像ファイル名).png に結果が保存されます。
        </p>
        <h3>露出不足の補正</h3>
        <p>
            python inv-dehaze-inv.py (画像ファイル名) [(weight：1～99：デフォルト70) (window_size：デフォルト10)]<br>
            <br>
            入力画像、inv-dehaze-invされた画像が表示されます。<br>
            ESCキー押下でプログラム終了。inv-dehaze-inv_(画像ファイル名).png に結果が保存されます。
        </p>
    </body>
</html>
