import cv2

# カメラデバイスをオープンします。デバイス番号は通常0です。
cap = cv2.VideoCapture(0)

# カメラが正常にオープンされたかチェックします。
if not cap.isOpened():
    print("カメラがオープンできませんでした。")
    exit()

while True:
    # フレームをキャプチャします。
    ret, frame = cap.read()

    # フレームのキャプチャが成功したかチェックします。
    if not ret:
        print("フレームをキャプチャできませんでした。")
        break

    # フレームを表示します。
    cv2.imshow("USB camera", frame)

    # 'q'キーを押すとループを終了します。
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラデバイスを解放します。
cap.release()

# ウィンドウを閉じます。
cv2.destroyAllWindows()