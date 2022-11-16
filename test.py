import time
import cv2
import numpy as np


def pause_key(keypress, seconds):
    key = 0
    print('准备开始，按空格键暂停及继续。。。。。。\n')
    img = np.zeros((100, 200, 3), np.uint8)
    img.fill(255)
    cv2.putText(img, '!!!!!!', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    cv2.imshow('attention!', img)
    for second in range(seconds):
        input_kb = cv2.waitKey(1) & 0xFF
        if input_kb == ord(' '):
            print('paused')
            cv2.waitKey(0)
            print('continued')
        time.sleep(1)
        print(second)
        second += 1
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pause_key(keypress=' ', seconds=10)