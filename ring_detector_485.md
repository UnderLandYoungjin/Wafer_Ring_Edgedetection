




https://github.com/user-attachments/assets/d80097e7-7476-49f3-8cab-19febb9e13c1

```python
# realtime_ring_modbus_plc_camera1.py
# USB 카메라 1번으로 금색 링을 검출하고 RS-485 Modbus RTU로 PLC D영역에 데이터를 쓰는 프로그램입니다.

import cv2  # OpenCV를 불러옵니다. 카메라 입력, 이미지 처리, 화면 출력에 사용합니다.
import numpy as np  # NumPy를 불러옵니다. 배열 계산과 HSV 마스크 처리에 사용합니다.
import time  # FPS 계산, PLC 전송 주기, 재연결 간격 측정에 사용합니다.
from typing import Optional, Dict, Any  # 함수 반환값과 측정 데이터 구조를 명확하게 표시하기 위해 사용합니다.

from pymodbus.client import ModbusSerialClient


CAMERA_INDEX = 1  # 사용할 USB 카메라 번호를 1번으로 지정합니다.
CAMERA_WIDTH = 1280  # 카메라에 요청할 가로 해상도를 지정합니다.
CAMERA_HEIGHT = 720  # 카메라에 요청할 세로 해상도를 지정합니다.
CAMERA_FPS = 30  # 카메라에 요청할 초당 프레임 수를 지정합니다.
PROCESS_WIDTH = 640  # 링 검출 연산을 수행할 내부 영상의 가로 크기를 지정합니다.

PLC_ENABLED = True  # True이면 PLC 통신을 사용하고 False이면 비전 기능만 실행합니다.
PLC_PORT = "COM3"  # USB-RS485 컨버터가 연결된 Windows COM 포트를 지정합니다.
PLC_BAUDRATE = 9600  # PLC와 동일한 Modbus RTU 통신 속도를 지정합니다.
PLC_BYTESIZE = 8  # PLC와 동일한 데이터 비트를 8비트로 지정합니다.
PLC_PARITY = "N"  # PLC와 동일하게 패리티 없음을 지정합니다.
PLC_STOPBITS = 1  # PLC와 동일하게 정지 비트를 1비트로 지정합니다.
PLC_TIMEOUT = 0.3  # PLC 응답을 기다릴 최대 시간을 초 단위로 지정합니다.
PLC_SLAVE_ID = 1  # PLC의 Modbus 국번 또는 Slave ID를 지정합니다.
PLC_D_BASE_ADDRESS = 0  # Modbus Holding Register 0번을 PLC D0에 대응시키는 기준 주소입니다.
PLC_SEND_INTERVAL_SEC = 0.2  # 측정 데이터를 PLC로 전송하는 최소 주기를 지정합니다.
PLC_RECONNECT_INTERVAL_SEC = 1.0  # PLC 연결 실패 후 재연결을 시도하는 최소 간격입니다.
PLC_NOT_FOUND_CONFIRM_FRAMES = 5  # 링 미검출이 이 프레임 수만큼 연속되어야 미검출 상태로 확정합니다.

D_STATUS_OFFSET = 0  # D0에는 검사 상태를 기록합니다. 0=정상 검출, 1=미검출, 2=통신 오류입니다.
D_OUTER_DIAMETER_OFFSET = 1  # D1에는 외경을 픽셀 단위 정수로 기록합니다.
D_INNER_DIAMETER_OFFSET = 2  # D2에는 추정 내경을 픽셀 단위 정수로 기록합니다.
D_THICKNESS_OFFSET = 3  # D3에는 추정 링 두께를 픽셀 단위 정수로 기록합니다.
D_CENTER_X_OFFSET = 4  # D4에는 링 중심의 X좌표를 픽셀 단위로 기록합니다.
D_CENTER_Y_OFFSET = 5  # D5에는 링 중심의 Y좌표를 픽셀 단위로 기록합니다.
D_HEARTBEAT_OFFSET = 6  # D6에는 프로그램이 동작 중임을 확인하는 생존 카운터를 기록합니다.

STATUS_OK = 0  # 링이 정상적으로 검출된 상태값입니다.
STATUS_NOT_FOUND = 1  # 링을 찾지 못한 상태값입니다.
STATUS_COMMUNICATION_ERROR = 2  # PLC 통신 오류를 표시할 때 사용하는 상태값입니다.


def nothing(value):  # Trackbar 값이 바뀔 때 OpenCV가 호출하는 함수입니다.
    pass  # Trackbar 변경 자체만 필요하므로 별도 동작은 수행하지 않습니다.


def draw_text(image, text, x, y, color=(0, 255, 255), scale=0.75):  # 화면에 상태 문자열을 표시하는 함수입니다.
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)  # 지정 위치에 문자열을 표시합니다.


def clamp_uint16(value):  # PLC 16비트 부호 없는 정수 범위에 맞게 값을 제한하는 함수입니다.
    return max(0, min(65535, int(value)))  # 입력값을 0 이상 65535 이하의 정수로 변환합니다.


def find_largest_contour(mask, min_area):  # 마스크에서 최소 면적 이상인 가장 큰 외곽선을 찾는 함수입니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외부 외곽선만 검출합니다.
    valid_contours = []  # 조건을 통과한 외곽선을 저장할 리스트를 만듭니다.

    for contour in contours:  # 검출된 모든 외곽선을 하나씩 확인합니다.
        area = cv2.contourArea(contour)  # 현재 외곽선의 면적을 계산합니다.
        if area >= min_area:  # 외곽선 면적이 최소 기준 이상인지 확인합니다.
            valid_contours.append(contour)  # 유효한 외곽선을 리스트에 추가합니다.

    if len(valid_contours) == 0:  # 유효한 외곽선이 하나도 없는지 확인합니다.
        return None  # 링 후보를 찾지 못했음을 반환합니다.

    return max(valid_contours, key=cv2.contourArea)  # 유효한 외곽선 중 면적이 가장 큰 것을 반환합니다.


class PlcModbusWriter:  # RS-485 Modbus RTU를 통해 PLC Holding Register에 값을 쓰는 클래스입니다.
    def __init__(self):  # PLC 통신 객체의 초기 상태를 설정합니다.
        self.client = None  # pymodbus 클라이언트 객체를 저장할 변수를 만듭니다.
        self.connected = False  # 현재 PLC 연결 상태를 False로 초기화합니다.
        self.last_error = ""  # 마지막 통신 오류 내용을 저장할 문자열을 만듭니다.
        self.last_connect_attempt = 0.0  # 마지막 연결 시도 시간을 저장합니다.
        self.last_send_time = 0.0  # 마지막 데이터 전송 시간을 저장합니다.
        self.last_values = {}  # 각 레지스터에 마지막으로 전송한 값을 저장합니다.

    def connect(self):  # PLC와 연결하거나 이미 연결된 상태를 확인하는 함수입니다.
        if not PLC_ENABLED:  # PLC 통신 기능이 꺼져 있는지 확인합니다.
            self.last_error = "PLC_DISABLED"  # PLC 기능이 꺼져 있다는 상태를 저장합니다.
            return False  # 연결하지 않았음을 반환합니다.

        now = time.time()  # 현재 시간을 가져옵니다.
        if now - self.last_connect_attempt < PLC_RECONNECT_INTERVAL_SEC:  # 재연결 제한 시간이 지나지 않았는지 확인합니다.
            return self.connected  # 기존 연결 상태를 그대로 반환합니다.

        self.last_connect_attempt = now  # 현재 시간을 마지막 연결 시도 시간으로 저장합니다.
        self.close()  # 남아 있을 수 있는 이전 직렬 포트 연결을 먼저 닫습니다.

        try:  # COM 포트 연결 과정에서 발생할 수 있는 예외를 처리합니다.
            self.client = ModbusSerialClient(  # pymodbus 직렬 클라이언트 객체를 생성합니다.
                port=PLC_PORT,  # USB-RS485 컨버터의 COM 포트를 지정합니다.
                baudrate=PLC_BAUDRATE,  # Modbus RTU 통신 속도를 지정합니다.
                bytesize=PLC_BYTESIZE,  # 데이터 비트를 지정합니다.
                parity=PLC_PARITY,  # 패리티 방식을 지정합니다.
                stopbits=PLC_STOPBITS,  # 정지 비트를 지정합니다.
                timeout=PLC_TIMEOUT,  # PLC 응답 제한 시간을 지정합니다.
            )  # ModbusSerialClient 생성 구문을 닫습니다.
            self.connected = bool(self.client.connect())  # 실제 COM 포트를 열고 성공 여부를 저장합니다.
            self.last_error = "" if self.connected else "PLC connect failed"  # 연결 실패 시 오류 문구를 저장합니다.

            if self.connected:  # PLC 연결에 성공했는지 확인합니다.
                print(f"[PLC] 연결 성공: {PLC_PORT}, {PLC_BAUDRATE}, 8-N-1, Slave={PLC_SLAVE_ID}")  # 연결 정보를 출력합니다.
            else:  # PLC 연결에 실패한 경우입니다.
                print(f"[PLC] 연결 실패: {self.last_error}")  # 연결 실패 내용을 출력합니다.

            return self.connected  # 최종 연결 성공 여부를 반환합니다.
        except Exception as error:  # COM 포트 점유나 설정 오류 등의 예외를 처리합니다.
            self.connected = False  # 연결 상태를 실패로 설정합니다.
            self.last_error = str(error)  # 발생한 예외 내용을 저장합니다.
            print(f"[PLC] 연결 예외: {self.last_error}")  # 예외 내용을 콘솔에 출력합니다.
            return False  # 연결 실패를 반환합니다.

    def close(self):  # PLC 직렬 통신 포트를 안전하게 닫는 함수입니다.
        try:  # 포트를 닫는 과정에서 발생할 수 있는 예외를 처리합니다.
            if self.client is not None:  # 생성된 클라이언트 객체가 있는지 확인합니다.
                self.client.close()  # pymodbus 직렬 포트를 닫습니다.
        except Exception:  # 종료 과정의 사소한 예외는 무시합니다.
            pass  # 별도 동작을 수행하지 않습니다.

        self.client = None  # 닫힌 클라이언트 객체 참조를 제거합니다.
        self.connected = False  # 연결 상태를 False로 갱신합니다.

    def _write_register_compatible(self, address, value):  # pymodbus 버전 차이를 처리하면서 레지스터 1개를 쓰는 함수입니다.
        try:  # 최신 pymodbus의 device_id 인자를 먼저 시도합니다.
            return self.client.write_register(address=address, value=value, device_id=PLC_SLAVE_ID)  # 최신 API로 FC06 쓰기를 실행합니다.
        except TypeError:  # 현재 버전에서 device_id 인자를 지원하지 않는 경우입니다.
            try:  # pymodbus 3.x 일부 버전에서 사용하는 slave 인자를 시도합니다.
                return self.client.write_register(address=address, value=value, slave=PLC_SLAVE_ID)  # slave 인자로 FC06 쓰기를 실행합니다.
            except TypeError:  # 현재 버전에서 slave 인자도 지원하지 않는 경우입니다.
                return self.client.write_register(address, value, unit=PLC_SLAVE_ID)  # pymodbus 2.x의 unit 인자로 FC06 쓰기를 실행합니다.

    def write_register(self, address, value, force=False):  # 지정한 Holding Register 한 개에 값을 쓰는 함수입니다.
        if not PLC_ENABLED:  # PLC 통신 기능이 꺼져 있는지 확인합니다.
            return False  # 실제 전송 없이 실패 상태를 반환합니다.

        value = clamp_uint16(value)  # 전송값을 PLC 16비트 범위로 제한합니다.
        now = time.time()  # 현재 시간을 가져옵니다.

        if not force and self.last_values.get(address) == value:  # 이전에 같은 주소에 같은 값을 보냈는지 확인합니다.
            return True  # 값이 바뀌지 않았으므로 불필요한 통신 없이 성공으로 처리합니다.

        if not self.connected:  # PLC가 연결되어 있지 않은지 확인합니다.
            if not self.connect():  # PLC 재연결을 시도하고 실패 여부를 확인합니다.
                return False  # 연결할 수 없으면 쓰기 작업을 종료합니다.

        try:  # Modbus FC06 쓰기 요청 중 발생할 수 있는 예외를 처리합니다.
            result = self._write_register_compatible(address, value)  # pymodbus 버전에 맞는 방식으로 레지스터를 씁니다.

            if result is None or result.isError():  # PLC가 오류 응답을 반환했는지 확인합니다.
                self.last_error = str(result)  # 오류 응답 내용을 저장합니다.
                self.connected = False  # 다음 호출에서 재연결하도록 연결 상태를 내립니다.
                print(f"[PLC] 쓰기 실패: Address={address}, Value={value}, Error={self.last_error}")  # 실패 정보를 출력합니다.
                return False  # 전송 실패를 반환합니다.

            self.last_values[address] = value  # 성공적으로 전송한 값을 주소별로 저장합니다.
            self.last_error = ""  # 이전 오류 내용을 초기화합니다.
            return True  # 전송 성공을 반환합니다.
        except Exception as error:  # 직렬 통신 예외나 포트 오류를 처리합니다.
            self.last_error = str(error)  # 예외 내용을 저장합니다.
            self.connected = False  # 다음 호출에서 재연결하도록 연결 상태를 내립니다.
            print(f"[PLC] 쓰기 예외: Address={address}, Value={value}, Error={self.last_error}")  # 예외 정보를 출력합니다.
            return False  # 전송 실패를 반환합니다.

    def write_measurements(self, status, measurements, heartbeat, force=False):  # 검사 결과 전체를 D0부터 D6까지 쓰는 함수입니다.
        now = time.time()  # 현재 시간을 가져옵니다.

        if not force and now - self.last_send_time < PLC_SEND_INTERVAL_SEC:  # 최소 전송 주기가 지나지 않았는지 확인합니다.
            return True  # 이번 프레임에서는 전송하지 않고 정상 처리합니다.

        self.last_send_time = now  # 현재 시간을 마지막 전송 시간으로 저장합니다.
        base = PLC_D_BASE_ADDRESS  # D영역에 대응하는 Modbus 기준 주소를 읽습니다.

        if measurements is None:  # 링 측정값이 없는 상태인지 확인합니다.
            values = {  # 미검출 시 PLC로 보낼 레지스터 값을 구성합니다.
                base + D_STATUS_OFFSET: status,  # D0에 미검출 상태를 기록합니다.
                base + D_OUTER_DIAMETER_OFFSET: 0,  # D1 외경값을 0으로 초기화합니다.
                base + D_INNER_DIAMETER_OFFSET: 0,  # D2 내경값을 0으로 초기화합니다.
                base + D_THICKNESS_OFFSET: 0,  # D3 두께값을 0으로 초기화합니다.
                base + D_CENTER_X_OFFSET: 0,  # D4 중심 X값을 0으로 초기화합니다.
                base + D_CENTER_Y_OFFSET: 0,  # D5 중심 Y값을 0으로 초기화합니다.
                base + D_HEARTBEAT_OFFSET: heartbeat,  # D6에 생존 카운터를 기록합니다.
            }  # 미검출 데이터 구성을 닫습니다.
        else:  # 링 측정값이 존재하는 경우입니다.
            values = {  # 정상 검출 시 PLC로 보낼 레지스터 값을 구성합니다.
                base + D_STATUS_OFFSET: status,  # D0에 정상 상태값을 기록합니다.
                base + D_OUTER_DIAMETER_OFFSET: measurements["outer_d"],  # D1에 외경 픽셀값을 기록합니다.
                base + D_INNER_DIAMETER_OFFSET: measurements["inner_d"],  # D2에 내경 추정 픽셀값을 기록합니다.
                base + D_THICKNESS_OFFSET: measurements["thickness"],  # D3에 두께 추정 픽셀값을 기록합니다.
                base + D_CENTER_X_OFFSET: measurements["center_x"],  # D4에 중심 X좌표를 기록합니다.
                base + D_CENTER_Y_OFFSET: measurements["center_y"],  # D5에 중심 Y좌표를 기록합니다.
                base + D_HEARTBEAT_OFFSET: heartbeat,  # D6에 생존 카운터를 기록합니다.
            }  # 정상 검출 데이터 구성을 닫습니다.

        all_success = True  # 전체 레지스터 쓰기 성공 여부를 True로 시작합니다.

        for address, value in values.items():  # 전송할 모든 주소와 값을 하나씩 확인합니다.
            success = self.write_register(address, value, force=force or address == base + D_HEARTBEAT_OFFSET)  # 각 레지스터를 FC06으로 씁니다.
            all_success = all_success and success  # 하나라도 실패하면 전체 결과가 False가 되도록 누적합니다.

        if all_success:  # 모든 레지스터 쓰기에 성공했는지 확인합니다.
            print(  # 한 번의 검사 결과 전송 내용을 콘솔에 출력합니다.
                f"[PLC] D{base}={status}, D{base + 1}={values[base + 1]}, "  # 상태와 외경 정보를 표시합니다.
                f"D{base + 2}={values[base + 2]}, D{base + 3}={values[base + 3]}, "  # 내경과 두께 정보를 표시합니다.
                f"D{base + 4}={values[base + 4]}, D{base + 5}={values[base + 5]}, "  # 중심 좌표 정보를 표시합니다.
                f"D{base + 6}={heartbeat}"  # 생존 카운터 정보를 표시합니다.
            )  # 출력 구문을 닫습니다.

        return all_success  # 전체 레지스터 전송 성공 여부를 반환합니다.


class RingDetectionState:  # 연속 미검출 프레임을 이용해 PLC 상태값을 안정화하는 클래스입니다.
    def __init__(self):  # 상태 판정에 필요한 변수를 초기화합니다.
        self.not_found_count = 0  # 연속 미검출 프레임 수를 0으로 초기화합니다.
        self.current_status = STATUS_NOT_FOUND  # 프로그램 시작 시 상태를 미검출로 초기화합니다.

    def update(self, measurements):  # 현재 프레임의 측정 결과를 받아 확정 상태를 계산합니다.
        if measurements is None:  # 현재 프레임에서 링을 찾지 못했는지 확인합니다.
            self.not_found_count += 1  # 연속 미검출 횟수를 1 증가시킵니다.

            if self.not_found_count >= PLC_NOT_FOUND_CONFIRM_FRAMES:  # 설정한 연속 미검출 기준에 도달했는지 확인합니다.
                self.current_status = STATUS_NOT_FOUND  # PLC 상태를 미검출로 확정합니다.
        else:  # 현재 프레임에서 링을 정상 검출한 경우입니다.
            self.not_found_count = 0  # 연속 미검출 횟수를 즉시 초기화합니다.
            self.current_status = STATUS_OK  # PLC 상태를 정상 검출로 설정합니다.

        return self.current_status  # 현재 확정된 검사 상태값을 반환합니다.


def detect_ring_keep_ratio(frame, debug_mode):  # 원본 비율을 유지하면서 금색 링을 검출하고 측정값을 반환하는 함수입니다.
    result = frame.copy()  # 원본 영상 위에 검출 결과를 표시하기 위해 복사본을 만듭니다.
    original_h, original_w = frame.shape[:2]  # 원본 영상의 높이와 너비를 가져옵니다.
    scale = PROCESS_WIDTH / original_w  # 원본에서 내부 처리 영상으로 줄이는 배율을 계산합니다.
    process_height = int(original_h * scale)  # 원본 비율을 유지하는 내부 처리 영상 높이를 계산합니다.
    small = cv2.resize(frame, (PROCESS_WIDTH, process_height))  # 원본 비율을 유지하면서 처리용 영상을 축소합니다.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)  # 금색을 분리하기 위해 BGR 영상을 HSV로 변환합니다.

    h_low = cv2.getTrackbarPos("H Low", "Control")  # Hue 하한 Trackbar 값을 읽습니다.
    h_high = cv2.getTrackbarPos("H High", "Control")  # Hue 상한 Trackbar 값을 읽습니다.
    s_low = cv2.getTrackbarPos("S Low", "Control")  # Saturation 하한 Trackbar 값을 읽습니다.
    v_low = cv2.getTrackbarPos("V Low", "Control")  # Value 하한 Trackbar 값을 읽습니다.

    lower_gold = np.array([h_low, s_low, v_low], dtype=np.uint8)  # 금색 HSV 하한값을 배열로 만듭니다.
    upper_gold = np.array([h_high, 255, 255], dtype=np.uint8)  # 금색 HSV 상한값을 배열로 만듭니다.
    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)  # HSV 범위에 해당하는 금색 영역만 흰색 마스크로 만듭니다.

    kernel = np.ones((5, 5), np.uint8)  # 형태학 연산에 사용할 5x5 커널을 만듭니다.
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, kernel, iterations=1)  # 작은 점 노이즈를 제거합니다.
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 끊어진 금색 영역을 연결합니다.
    contour = find_largest_contour(gold_mask, 800)  # 금색 마스크에서 가장 큰 링 후보 외곽선을 찾습니다.
    measurements: Optional[Dict[str, int]] = None  # PLC로 보낼 측정값을 저장할 변수를 초기화합니다.

    if contour is not None:  # 유효한 금색 링 후보를 찾았는지 확인합니다.
        contour_original = (contour / scale).astype(np.int32)  # 축소 영상 좌표를 원본 영상 좌표로 환산합니다.
        cv2.drawContours(result, [contour_original], -1, (0, 255, 0), 2)  # 실제 금색 외곽선을 초록색으로 표시합니다.

        (ox_float, oy_float), outer_r_float = cv2.minEnclosingCircle(contour_original)  # 외곽선을 감싸는 최소 외접원을 계산합니다.
        ox = int(round(ox_float))  # 외접원 중심 X좌표를 정수로 변환합니다.
        oy = int(round(oy_float))  # 외접원 중심 Y좌표를 정수로 변환합니다.
        outer_r = int(round(outer_r_float))  # 외접원 반지름을 정수로 변환합니다.
        inner_r = int(round(outer_r * 0.62))  # 기존 코드와 동일하게 내경 반지름을 외경 비율로 임시 추정합니다.
        outer_d = outer_r * 2  # 외경을 픽셀 단위로 계산합니다.
        inner_d = inner_r * 2  # 내경 추정값을 픽셀 단위로 계산합니다.
        thickness = outer_r - inner_r  # 링 한쪽 두께의 추정값을 픽셀 단위로 계산합니다.

        measurements = {  # PLC 전송과 화면 표시에서 사용할 측정 데이터 사전을 만듭니다.
            "outer_d": clamp_uint16(outer_d),  # 외경값을 16비트 범위로 저장합니다.
            "inner_d": clamp_uint16(inner_d),  # 내경 추정값을 16비트 범위로 저장합니다.
            "thickness": clamp_uint16(thickness),  # 두께 추정값을 16비트 범위로 저장합니다.
            "center_x": clamp_uint16(ox),  # 중심 X좌표를 16비트 범위로 저장합니다.
            "center_y": clamp_uint16(oy),  # 중심 Y좌표를 16비트 범위로 저장합니다.
        }  # 측정 데이터 사전 생성을 닫습니다.

        cv2.circle(result, (ox, oy), outer_r, (255, 0, 0), 2)  # 외경 기준 원을 파란색으로 표시합니다.
        cv2.circle(result, (ox, oy), inner_r, (0, 255, 255), 2)  # 추정 내경 원을 노란색으로 표시합니다.
        cv2.circle(result, (ox, oy), 4, (0, 0, 255), -1)  # 링 중심점을 빨간색으로 표시합니다.

        draw_text(result, f"Outer D: {outer_d}px", 20, 35)  # 화면에 외경 픽셀값을 표시합니다.
        draw_text(result, f"Inner D approx: {inner_d}px", 20, 70)  # 화면에 내경 추정 픽셀값을 표시합니다.
        draw_text(result, f"Thickness approx: {thickness}px", 20, 105)  # 화면에 두께 추정 픽셀값을 표시합니다.
        draw_text(result, f"Center: ({ox}, {oy})", 20, 140)  # 화면에 중심 좌표를 표시합니다.
    else:  # 금색 링 후보를 찾지 못한 경우입니다.
        draw_text(result, "Ring Not Found", 20, 35, color=(0, 0, 255))  # 미검출 메시지를 빨간색으로 표시합니다.

    if debug_mode:  # 사용자가 디버그 화면을 켰는지 확인합니다.
        mask_bgr = cv2.cvtColor(gold_mask, cv2.COLOR_GRAY2BGR)  # 흑백 금색 마스크를 3채널 영상으로 변환합니다.
        mask_show = cv2.resize(mask_bgr, (original_w, original_h))  # 마스크를 원본 영상 크기로 확대합니다.
        debug_view = np.hstack([result, mask_show])  # 검출 화면과 마스크 화면을 좌우로 결합합니다.
        return debug_view, measurements  # 디버그 화면과 측정값을 함께 반환합니다.

    return result, measurements  # 일반 검출 화면과 측정값을 함께 반환합니다.


def main():  # 카메라, 비전 검사, PLC 통신을 실행하는 메인 함수입니다.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows DirectShow 방식으로 USB 카메라 1번을 엽니다.

    if not cap.isOpened():  # 카메라가 정상적으로 열리지 않았는지 확인합니다.
        print(f"USB 카메라 {CAMERA_INDEX}번을 열 수 없습니다.")  # 카메라 연결 실패 메시지를 출력합니다.
        return  # 프로그램 실행을 종료합니다.

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 높은 FPS를 위해 MJPG 압축 포맷을 요청합니다.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)  # 카메라 가로 해상도를 요청합니다.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)  # 카메라 세로 해상도를 요청합니다.
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)  # 카메라 FPS를 요청합니다.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 카메라 버퍼를 줄여 영상 지연을 줄입니다.

    cv2.namedWindow("Control")  # HSV 검출값을 조정할 컨트롤 창을 만듭니다.
    cv2.createTrackbar("H Low", "Control", 15, 179, nothing)  # 금색 Hue 하한 Trackbar를 만듭니다.
    cv2.createTrackbar("H High", "Control", 45, 179, nothing)  # 금색 Hue 상한 Trackbar를 만듭니다.
    cv2.createTrackbar("S Low", "Control", 40, 255, nothing)  # 금색 채도 하한 Trackbar를 만듭니다.
    cv2.createTrackbar("V Low", "Control", 60, 255, nothing)  # 금색 밝기 하한 Trackbar를 만듭니다.

    plc = PlcModbusWriter()  # PLC Modbus RTU 통신 객체를 생성합니다.
    plc.connect()  # 프로그램 시작 시 PLC 연결을 한 번 시도합니다.
    ring_state = RingDetectionState()  # 링 검출 상태 안정화 객체를 생성합니다.

    prev_time = time.time()  # FPS 계산을 위한 이전 프레임 시간을 저장합니다.
    debug_mode = False  # 디버그 화면 표시 상태를 끈 상태로 시작합니다.
    save_count = 0  # 저장 이미지 번호를 0으로 초기화합니다.
    heartbeat = 0  # PLC에서 프로그램 생존 여부를 확인할 카운터를 0으로 초기화합니다.

    print("USB Camera Ring Detection + PLC Modbus RTU 프로그램 시작")  # 프로그램 시작 메시지를 출력합니다.
    print(f"Camera Index: {CAMERA_INDEX}, Resolution Request: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")  # 카메라 설정을 출력합니다.
    print(f"PLC: {PLC_PORT}, {PLC_BAUDRATE}, 8-N-1, Slave ID={PLC_SLAVE_ID}")  # PLC 통신 설정을 출력합니다.
    print(f"PLC Register Map: D{PLC_D_BASE_ADDRESS}~D{PLC_D_BASE_ADDRESS + 6}")  # 사용하는 PLC D영역 범위를 출력합니다.
    print("Keys: D=Debug, S=Save, Q or ESC=Exit")  # 프로그램 조작키를 출력합니다.

    try:  # 프로그램 종료 시 카메라와 PLC 포트를 반드시 닫기 위해 try 블록을 시작합니다.
        while True:  # 사용자가 종료할 때까지 카메라 검사를 반복합니다.
            ret, frame = cap.read()  # USB 카메라에서 한 프레임을 읽습니다.

            if not ret:  # 카메라 프레임 읽기에 실패했는지 확인합니다.
                print("카메라 프레임을 읽을 수 없습니다.")  # 프레임 읽기 실패 메시지를 출력합니다.
                break  # 메인 반복문을 종료합니다.

            view, measurements = detect_ring_keep_ratio(frame, debug_mode)  # 금색 링을 검출하고 측정값을 계산합니다.
            status = ring_state.update(measurements)  # 연속 미검출 조건을 반영한 확정 상태를 계산합니다.
            heartbeat = (heartbeat + 1) % 65536  # 생존 카운터를 0부터 65535 범위에서 순환 증가시킵니다.
            plc_success = plc.write_measurements(status, measurements, heartbeat)  # D0부터 D6까지 검사 데이터를 전송합니다.

            now_time = time.time()  # 현재 시간을 가져옵니다.
            fps = 1.0 / max(now_time - prev_time, 0.0001)  # 현재 프레임 처리 FPS를 계산합니다.
            prev_time = now_time  # 현재 시간을 다음 프레임 계산 기준으로 저장합니다.

            status_text = "OK" if status == STATUS_OK else "NOT FOUND"  # 상태값을 화면 표시용 문자열로 변환합니다.
            plc_text = "CONNECTED" if plc.connected and plc_success else f"ERROR: {plc.last_error}"  # PLC 연결 상태 문자열을 만듭니다.
            status_color = (0, 255, 0) if status == STATUS_OK else (0, 0, 255)  # 검사 상태에 따라 표시 색상을 결정합니다.
            plc_color = (0, 255, 0) if plc.connected and plc_success else (0, 0, 255)  # PLC 상태에 따라 표시 색상을 결정합니다.

            draw_text(view, f"Inspection: {status_text}", 20, view.shape[0] - 110, color=status_color)  # 검사 상태를 화면에 표시합니다.
            draw_text(view, f"PLC: {plc_text}", 20, view.shape[0] - 80, color=plc_color, scale=0.60)  # PLC 상태를 화면에 표시합니다.
            draw_text(view, f"FPS: {fps:.1f}", 20, view.shape[0] - 50)  # 현재 처리 FPS를 화면에 표시합니다.
            draw_text(view, f"Camera Index: {CAMERA_INDEX}", 20, view.shape[0] - 20)  # 사용 중인 카메라 번호를 표시합니다.

            cv2.imshow("Realtime Ring Measurement + PLC Modbus RTU", view)  # 최종 결과 화면을 표시합니다.
            key = cv2.waitKey(1) & 0xFF  # 키보드 입력을 1밀리초 동안 기다립니다.

            if key == ord("d") or key == ord("D"):  # 사용자가 D키를 눌렀는지 확인합니다.
                debug_mode = not debug_mode  # 검출 화면과 마스크를 함께 보는 디버그 모드를 전환합니다.

            if key == ord("s") or key == ord("S"):  # 사용자가 S키를 눌렀는지 확인합니다.
                filename = f"ring_plc_result_{save_count}.png"  # 저장할 이미지 파일명을 만듭니다.
                cv2.imwrite(filename, view)  # 현재 결과 화면을 PNG 파일로 저장합니다.
                print(f"저장 완료: {filename}")  # 저장된 파일명을 콘솔에 출력합니다.
                save_count += 1  # 다음 이미지 저장 번호를 증가시킵니다.

            if key == 27 or key == ord("q") or key == ord("Q"):  # ESC키 또는 Q키가 눌렸는지 확인합니다.
                break  # 메인 반복문을 종료합니다.
    finally:  # 정상 종료와 예외 종료 모두에서 반드시 실행되는 정리 구문입니다.
        plc.close()  # PLC Modbus RTU 직렬 포트를 닫습니다.
        cap.release()  # USB 카메라 장치를 해제합니다.
        cv2.destroyAllWindows()  # 프로그램이 생성한 OpenCV 창을 모두 닫습니다.
        print("프로그램 종료: 카메라와 PLC 포트를 해제했습니다.")  # 자원 해제 완료 메시지를 출력합니다.


if __name__ == "__main__":  # 이 파일을 직접 실행한 경우에만 메인 함수를 호출합니다.
    main()  # 카메라 검사와 PLC 통신 프로그램을 시작합니다.



```
