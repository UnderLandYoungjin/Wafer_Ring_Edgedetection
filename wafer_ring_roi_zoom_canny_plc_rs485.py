# C:\py_code\wafer_ring_roi_zoom_canny.py  # Parking 카메라에서 모따기 영역만 잘라 확대하고 가로 엣지를 검출하며 두 경계선 사이 거리를 측정/시각화하는 코드입니다.

import os  # RTSP 옵션 설정과 폴더 생성을 위해 사용합니다.
import time  # 재연결 대기와 FPS 계산을 위해 사용합니다.
import socket  # 카메라 IP/포트 연결 가능 여부 확인에 사용합니다.
import threading  # 카메라 영상을 별도 스레드에서 읽기 위해 사용합니다.
from datetime import datetime  # 스냅샷 저장 파일명에 시간을 넣기 위해 사용합니다.
from urllib.parse import urlparse  # RTSP 주소에서 IP와 포트를 추출하기 위해 사용합니다.

import cv2  # OpenCV 영상 처리 라이브러리입니다.
import numpy as np  # 이미지 배열 생성과 처리를 위해 사용합니다.

try:  # RS-485 Modbus 통신용 라이브러리 import를 시도합니다.
    from pymodbus.client import ModbusSerialClient  # pymodbus 3.x/4.x에서 사용하는 직렬 Modbus RTU 클라이언트입니다.
except ImportError:  # pymodbus가 설치되어 있지 않으면 실행됩니다.
    ModbusSerialClient = None  # 라이브러리가 없을 때 프로그램이 바로 죽지 않도록 None으로 둡니다.

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000|max_delay;500000"  # RTSP를 TCP 방식으로 받고 지연과 타임아웃을 줄입니다.

CAMERA_NAME = "Parking"  # 사용할 카메라 이름입니다.
CAMERA_URL = "rtsp://192.168.1.35/h264"  # 사용할 Parking 카메라 RTSP 주소입니다.

VIEW_W = 640  # 원본 표시 영상의 가로 크기입니다.
VIEW_H = 480  # 원본 표시 영상의 세로 크기입니다.

ROI_X1 = 0  # 검사 영역의 왼쪽 X좌표입니다.
ROI_Y1 = 210  # 검사 영역의 위쪽 Y좌표입니다.
ROI_X2 = 640  # 검사 영역의 오른쪽 X좌표입니다.
ROI_Y2 = 330  # 검사 영역의 아래쪽 Y좌표입니다.

ZOOM_SCALE = 3.0  # ROI를 몇 배 확대할지 정합니다.
BLUR_KERNEL = 7  # 표면 스크래치 잡음을 줄이기 위한 Gaussian Blur 커널 크기입니다. 키우면 잡음이 더 줄지만 약한 엣지도 뭉갤 수 있습니다.

CANNY_THRESHOLD_1 = 60  # 참고용으로 흐리게 표시할 Canny Edge 낮은 임계값입니다.
CANNY_THRESHOLD_2 = 160  # 참고용으로 흐리게 표시할 Canny Edge 높은 임계값입니다.

SOBEL_KERNEL = 3  # 세로 방향 그래디언트 계산용 Sobel 미분 커널 크기입니다.
HORIZONTAL_GRAD_THRESHOLD = 30  # 가로 엣지로 인정할 세로 그래디언트 절대 세기입니다. 완만한 경계도 잡히도록 낮게 잡습니다. 잡음이 많으면 키우세요.
HORIZONTAL_DOMINANCE = 1.1  # 세로 그래디언트가 가로 그래디언트보다 이 배수 이상 커야 가로 엣지로 인정합니다.
HORIZONTAL_MORPH_W = 25  # 끊긴 가로 엣지를 이어 붙이는 가로 방향 커널의 길이입니다.
HORIZONTAL_MORPH_H = 3  # 가로 방향 커널의 세로 두께입니다.
MIN_EDGE_WIDTH = 200  # 이 값보다 가로로 짧은 엣지 덩어리는 스크래치 잡음으로 보고 제거합니다. 경계선만 남기고 싶을 때 키웁니다.

MEASURE_ENABLED = True  # 초록색 가로 엣지 2개 사이의 거리를 측정할지 여부입니다.
MEASURE_X1 = 120  # 확대 ROI 화면 기준 측정 박스의 왼쪽 X좌표입니다. 실제 화면에서 빨간 박스 위치를 이 값으로 조정합니다.
MEASURE_Y1 = 120  # 확대 ROI 화면 기준 측정 박스의 위쪽 Y좌표입니다. 실제 화면에서 빨간 박스 위치를 이 값으로 조정합니다.
MEASURE_X2 = 420  # 확대 ROI 화면 기준 측정 박스의 오른쪽 X좌표입니다. 두 경계가 모두 포함되도록 조정합니다.
MEASURE_Y2 = 320  # 확대 ROI 화면 기준 측정 박스의 아래쪽 Y좌표입니다. 두 경계가 모두 포함되도록 조정합니다.
ROW_MIN_PIXELS = 25  # 한 행에서 초록 엣지 픽셀이 이 개수 이상일 때만 실제 경계 후보로 인정합니다.
ROW_CLUSTER_GAP = 8  # 같은 경계로 묶을 행 간격입니다. 경계선이 두껍거나 끊길 때 조금 키웁니다.
PIXEL_TO_MM = None  # 확대 영상 1픽셀이 몇 mm인지 알면 예: 0.005 처럼 입력합니다. 모르면 None으로 두면 px만 표시합니다.

# ---------------- 측정 시각화 색상(BGR) ----------------  # 확대 실영상과 엣지 영상 양쪽에서 잘 보이도록 고른 색상 구간입니다.
MEASURE_BOX_COLOR = (0, 0, 255)  # 측정 박스 색상(빨강)입니다.
MEASURE_LINE_COLOR = (255, 255, 0)  # 위/아래 경계 기준선 색상(시안)입니다. 회색 금속면과 초록 엣지 위에서 잘 보입니다.
MEASURE_ARROW_COLOR = (255, 0, 255)  # 거리 화살표 색상(마젠타)입니다. 초록/시안과 확실히 구분됩니다.
MEASURE_FILL_COLOR = (255, 255, 0)  # 두 경계 사이를 반투명하게 채울 색상(시안)입니다.
MEASURE_FILL_ALPHA = 0.22  # 반투명 채움의 진하기입니다. 0에 가까울수록 옅어집니다.
MEASURE_TEXT_COLOR = (0, 255, 255)  # 측정값 글자 색상(노랑)입니다.
# -------------------------------------------------------  # 측정 시각화 색상 구간의 끝입니다.

GRID_ENABLED = True  # 프로그램 시작 시 격자를 표시할지 여부입니다. 실행 중 [g] 키로 켜고 끌 수 있습니다.
GRID_SPACING = 100  # 격자 눈금 간격(픽셀)입니다. 값을 줄이면 눈금이 촘촘해집니다.
GRID_COLOR = (128, 128, 128)  # 격자 선 색상(BGR)입니다.
GRID_THICKNESS = 1  # 격자 선 두께입니다.
GRID_LABEL = True  # 격자 선에 픽셀 좌표 숫자를 표시할지 여부입니다.


# ---------------- PLC RS-485 Modbus RTU 출력 설정 ----------------  # Distance NOT FOUND 상태를 PLC D영역으로 보내기 위한 설정입니다.
PLC_ENABLED = True  # PLC 전송 기능을 사용할지 여부입니다. 처음 테스트가 불안하면 False로 두고 화면만 확인하세요.
PLC_PORT = "COM3"  # USB-RS485 컨버터가 잡힌 윈도우 COM 포트입니다. 장치 관리자에서 실제 COM 번호로 바꾸세요.
PLC_BAUDRATE = 9600  # PLC 통신 속도입니다. PLC 채널 설정과 반드시 같아야 합니다.
PLC_BYTESIZE = 8  # 데이터 비트입니다. 일반적인 Modbus RTU 설정은 8입니다.
PLC_PARITY = "N"  # 패리티입니다. N=None, E=Even, O=Odd이며 PLC 설정과 반드시 같아야 합니다.
PLC_STOPBITS = 1  # 정지 비트입니다. PLC 설정과 반드시 같아야 합니다.
PLC_TIMEOUT = 0.3  # PLC 응답 대기 시간입니다. 너무 짧으면 통신 실패가 늘어날 수 있습니다.
PLC_SLAVE_ID = 1  # PLC Modbus 국번입니다. PLC의 Slave ID/Station No.와 맞추세요.
PLC_D_REGISTER_ADDRESS = 0  # 쓸 D영역 주소입니다. 예: D0에 쓰려면 보통 0, D100에 쓰려면 보통 100입니다. PLC 매뉴얼에서 확인하세요.
PLC_VALUE_OK = 0  # 정상 측정 상태일 때 D영역에 쓸 값입니다.
PLC_VALUE_NOT_FOUND = 1  # Distance NOT FOUND 상태일 때 D영역에 쓸 값입니다.
PLC_SEND_OK_STATE = True  # NOT FOUND가 풀렸을 때 정상값 0도 PLC로 보낼지 여부입니다.
PLC_NOT_FOUND_CONFIRM_FRAMES = 1  # NOT FOUND가 이 프레임 수만큼 연속 발생해야 PLC에 1을 씁니다. 순간 오검출 방지용입니다.
PLC_MIN_SEND_INTERVAL_SEC = 0.2  # PLC에 같은 값을 너무 자주 쓰지 않기 위한 최소 전송 간격입니다.
# ---------------------------------------------------------------  # PLC 설정 구간의 끝입니다.

RECONNECT_DELAY = 3.0  # 연결 실패 시 재연결 전 대기 시간입니다.
PORT_CHECK_TIMEOUT = 3.0  # 카메라 IP와 포트 확인 제한 시간입니다.
SNAPSHOT_DIR = "snapshots"  # 스냅샷 저장 폴더입니다.
WINDOW_NAME = "Wafer Ring ROI Zoom Canny"  # OpenCV 창 이름입니다.
FONT = cv2.FONT_HERSHEY_SIMPLEX  # 화면 글자 출력용 폰트입니다.


class PlcModbusWriter:  # Distance 상태를 RS-485 Modbus RTU로 PLC D영역에 쓰는 클래스입니다.
    def __init__(self):  # PLC 통신 객체를 초기화합니다.
        self.client = None  # pymodbus 클라이언트 객체를 저장할 변수입니다.
        self.connected = False  # PLC 연결 성공 여부를 저장합니다.
        self.last_value = None  # 마지막으로 PLC에 전송한 값을 저장합니다.
        self.last_send_time = 0.0  # 마지막 전송 시각을 저장합니다.
        self.last_error = ""  # 마지막 통신 오류 문구를 저장합니다.

    def connect(self):  # PLC와 Modbus RTU 연결을 시도하는 함수입니다.
        if not PLC_ENABLED:  # PLC 기능이 꺼져 있으면 연결하지 않습니다.
            self.connected = False  # 연결 상태를 False로 둡니다.
            self.last_error = "PLC DISABLED"  # PLC 기능 비활성 상태를 기록합니다.
            return False  # 연결하지 않았음을 반환합니다.
        if ModbusSerialClient is None:  # pymodbus 라이브러리가 설치되지 않았는지 확인합니다.
            self.connected = False  # 연결 상태를 False로 둡니다.
            self.last_error = "pymodbus not installed"  # 설치 필요 문구를 기록합니다.
            return False  # 연결 실패를 반환합니다.
        try:  # 포트 연결 중 예외가 날 수 있어 보호합니다.
            self.client = ModbusSerialClient(  # Modbus RTU 직렬 클라이언트를 생성합니다.
                port=PLC_PORT,  # USB-RS485 컨버터의 COM 포트를 지정합니다.
                baudrate=PLC_BAUDRATE,  # PLC와 동일한 통신 속도를 지정합니다.
                bytesize=PLC_BYTESIZE,  # PLC와 동일한 데이터 비트를 지정합니다.
                parity=PLC_PARITY,  # PLC와 동일한 패리티를 지정합니다.
                stopbits=PLC_STOPBITS,  # PLC와 동일한 정지 비트를 지정합니다.
                timeout=PLC_TIMEOUT,  # PLC 응답 대기 시간을 지정합니다.
            )  # 클라이언트 생성 구문을 닫습니다.
            self.connected = bool(self.client.connect())  # 실제 포트를 열고 연결 성공 여부를 저장합니다.
            self.last_error = "" if self.connected else "connect failed"  # 실패 시 오류 문구를 기록합니다.
            return self.connected  # 연결 성공 여부를 반환합니다.
        except Exception as e:  # COM 포트 오류 등 예외를 처리합니다.
            self.connected = False  # 연결 실패 상태로 둡니다.
            self.last_error = str(e)  # 예외 메시지를 저장합니다.
            return False  # 연결 실패를 반환합니다.

    def close(self):  # PLC 통신 포트를 닫는 함수입니다.
        try:  # 닫는 과정에서도 예외가 날 수 있어 보호합니다.
            if self.client is not None:  # 클라이언트 객체가 존재하는지 확인합니다.
                self.client.close()  # 직렬 포트를 닫습니다.
        except Exception:  # 닫기 오류는 종료 과정에서 무시합니다.
            pass  # 별도 처리를 하지 않습니다.
        self.connected = False  # 연결 상태를 False로 갱신합니다.

    def write_d(self, value: int, force: bool = False) -> bool:  # PLC D영역에 정수 1개를 쓰는 함수입니다.
        if not PLC_ENABLED:  # PLC 기능이 꺼져 있으면 전송하지 않습니다.
            return False  # 전송하지 않았음을 반환합니다.
        now = time.time()  # 현재 시간을 가져옵니다.
        if not force and self.last_value == value and now - self.last_send_time < PLC_MIN_SEND_INTERVAL_SEC:  # 같은 값을 너무 자주 보내는지 확인합니다.
            return True  # 이미 같은 상태를 보냈으므로 성공으로 간주합니다.
        if not self.connected:  # 연결되어 있지 않으면 재연결을 시도합니다.
            if not self.connect():  # 재연결이 실패했는지 확인합니다.
                print(f"[PLC] 연결 실패: {self.last_error}")  # 연결 실패 원인을 출력합니다.
                return False  # 전송 실패를 반환합니다.
        try:  # Modbus 쓰기 중 예외가 날 수 있어 보호합니다.
            result = self.client.write_register(  # Holding Register 1개에 값을 씁니다.
                address=PLC_D_REGISTER_ADDRESS,  # PLC D영역에 매핑된 Modbus 주소를 지정합니다.
                value=int(value),  # PLC에 쓸 정수 값을 지정합니다.
                slave=PLC_SLAVE_ID,  # PLC 국번을 지정합니다.
            )  # 쓰기 요청 구문을 닫습니다.
            if result is None or result.isError():  # PLC가 오류 응답을 보냈는지 확인합니다.
                self.last_error = str(result)  # 오류 내용을 저장합니다.
                print(f"[PLC] 쓰기 실패: D{PLC_D_REGISTER_ADDRESS}={value}, error={self.last_error}")  # 쓰기 실패를 출력합니다.
                self.connected = False  # 다음 루프에서 재연결하도록 연결 상태를 내립니다.
                return False  # 전송 실패를 반환합니다.
            self.last_value = int(value)  # 마지막 전송 값을 저장합니다.
            self.last_send_time = now  # 마지막 전송 시간을 저장합니다.
            self.last_error = ""  # 오류 문구를 초기화합니다.
            print(f"[PLC] 쓰기 성공: D{PLC_D_REGISTER_ADDRESS}={value}")  # 쓰기 성공 로그를 출력합니다.
            return True  # 전송 성공을 반환합니다.
        except TypeError:  # pymodbus 구버전에서 slave 대신 unit 인자를 쓰는 경우를 처리합니다.
            try:  # 구버전 API 호출도 예외 보호합니다.
                result = self.client.write_register(PLC_D_REGISTER_ADDRESS, int(value), unit=PLC_SLAVE_ID)  # 구버전 방식으로 Holding Register에 값을 씁니다.
                if result is None or result.isError():  # 오류 응답 여부를 확인합니다.
                    self.last_error = str(result)  # 오류 내용을 저장합니다.
                    self.connected = False  # 다음 루프에서 재연결하도록 연결 상태를 내립니다.
                    return False  # 전송 실패를 반환합니다.
                self.last_value = int(value)  # 마지막 전송 값을 저장합니다.
                self.last_send_time = now  # 마지막 전송 시간을 저장합니다.
                self.last_error = ""  # 오류 문구를 초기화합니다.
                print(f"[PLC] 쓰기 성공: D{PLC_D_REGISTER_ADDRESS}={value}")  # 쓰기 성공 로그를 출력합니다.
                return True  # 전송 성공을 반환합니다.
            except Exception as e:  # 구버전 방식도 실패한 경우입니다.
                self.last_error = str(e)  # 예외 메시지를 저장합니다.
                self.connected = False  # 연결 상태를 False로 둡니다.
                print(f"[PLC] 쓰기 예외: {self.last_error}")  # 예외 내용을 출력합니다.
                return False  # 전송 실패를 반환합니다.
        except Exception as e:  # 일반적인 통신 예외를 처리합니다.
            self.last_error = str(e)  # 예외 메시지를 저장합니다.
            self.connected = False  # 다음 루프에서 재연결하도록 연결 상태를 내립니다.
            print(f"[PLC] 쓰기 예외: {self.last_error}")  # 예외 내용을 출력합니다.
            return False  # 전송 실패를 반환합니다.


class DistanceNotFoundState:  # Distance NOT FOUND 상태를 안정적으로 판정하고 PLC에 보낼 값을 결정하는 클래스입니다.
    def __init__(self):  # 상태 관리 변수를 초기화합니다.
        self.not_found_count = 0  # NOT FOUND가 연속으로 발생한 프레임 수입니다.
        self.current_state = PLC_VALUE_OK  # 현재 확정 상태값입니다. 0=정상, 1=NOT FOUND입니다.

    def update(self, distance_result) -> int | None:  # 거리 측정 결과를 받아 PLC로 보낼 새 값이 있는지 판단합니다.
        if distance_result is None:  # 현재 프레임이 NOT FOUND인지 확인합니다.
            self.not_found_count += 1  # 연속 NOT FOUND 카운트를 증가시킵니다.
            if self.not_found_count >= PLC_NOT_FOUND_CONFIRM_FRAMES and self.current_state != PLC_VALUE_NOT_FOUND:  # 확정 프레임 수를 넘고 기존 상태가 1이 아니면 전송 대상입니다.
                self.current_state = PLC_VALUE_NOT_FOUND  # 현재 상태를 NOT FOUND로 확정합니다.
                return PLC_VALUE_NOT_FOUND  # PLC에 1을 보내도록 반환합니다.
            return None  # 아직 확정 전이거나 이미 보낸 상태라 전송하지 않습니다.
        self.not_found_count = 0  # 정상 측정이 되면 NOT FOUND 카운트를 초기화합니다.
        if PLC_SEND_OK_STATE and self.current_state != PLC_VALUE_OK:  # 이전 상태가 NOT FOUND였고 정상 복귀값 전송이 켜져 있는지 확인합니다.
            self.current_state = PLC_VALUE_OK  # 현재 상태를 정상으로 확정합니다.
            return PLC_VALUE_OK  # PLC에 0을 보내도록 반환합니다.
        self.current_state = PLC_VALUE_OK  # 정상 상태를 유지합니다.
        return None  # 새로 보낼 값이 없습니다.


def check_port(ip: str, port: int, timeout: float) -> bool:  # IP와 포트 연결 가능 여부를 확인하는 함수입니다.
    try:  # 연결 시도 중 오류 발생 가능성이 있어 예외 처리를 시작합니다.
        with socket.create_connection((ip, port), timeout=timeout):  # 지정한 IP와 포트로 TCP 연결을 시도합니다.
            return True  # 연결 성공 시 True를 반환합니다.
    except OSError:  # 연결 실패나 네트워크 오류가 발생하면 실행됩니다.
        return False  # 연결 실패 시 False를 반환합니다.


class CameraStream:  # RTSP 카메라 1대를 관리하는 클래스입니다.
    def __init__(self, name: str, url: str):  # 카메라 이름과 RTSP 주소를 받아 초기화합니다.
        self.name = name  # 카메라 이름을 저장합니다.
        self.url = url  # RTSP 주소를 저장합니다.
        parsed = urlparse(url)  # RTSP 주소를 분석합니다.
        self.ip = parsed.hostname or ""  # RTSP 주소에서 IP를 가져옵니다.
        self.port = parsed.port or 554  # RTSP 주소에 포트가 없으면 기본 RTSP 포트 554를 사용합니다.
        self._frame = None  # 최신 프레임을 저장할 변수입니다.
        self._lock = threading.Lock()  # 프레임 접근 충돌을 막기 위한 잠금 객체입니다.
        self._running = False  # 스레드 실행 상태입니다.
        self._thread = None  # 카메라 수신 스레드 객체입니다.
        self.connected = False  # 카메라 연결 상태입니다.
        self.status = "WAIT"  # 현재 상태 문구입니다.
        self.fps = 0.0  # 수신 FPS 값입니다.
        self._fc = 0  # FPS 계산용 프레임 카운터입니다.
        self._t0 = time.time()  # FPS 계산 기준 시간입니다.

    def start(self):  # 카메라 수신 스레드를 시작하는 함수입니다.
        self._running = True  # 실행 상태를 True로 설정합니다.
        self._thread = threading.Thread(target=self._loop, daemon=True)  # 백그라운드 수신 스레드를 생성합니다.
        self._thread.start()  # 수신 스레드를 시작합니다.

    def _open(self) -> cv2.VideoCapture:  # RTSP 스트림을 여는 함수입니다.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)  # FFmpeg 백엔드로 RTSP 영상을 엽니다.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지연을 줄이기 위해 버퍼를 1로 설정합니다.
        return cap  # VideoCapture 객체를 반환합니다.

    def _loop(self):  # 카메라 영상을 계속 읽는 내부 루프입니다.
        while self._running:  # 프로그램이 실행 중인 동안 반복합니다.
            if not check_port(self.ip, self.port, PORT_CHECK_TIMEOUT):  # IP와 포트 연결 가능 여부를 확인합니다.
                self.connected = False  # 연결 상태를 False로 설정합니다.
                self.status = "NO ROUTE IP/PORT"  # IP 또는 포트 문제 상태로 표시합니다.
                time.sleep(RECONNECT_DELAY)  # 재연결 전 잠시 대기합니다.
                continue  # 루프 처음으로 돌아갑니다.

            cap = self._open()  # RTSP 스트림을 엽니다.
            if not cap.isOpened():  # 스트림 열기 실패 여부를 확인합니다.
                self.connected = False  # 연결 상태를 False로 설정합니다.
                self.status = "OPEN FAIL PATH/AUTH"  # 경로 또는 인증 문제 상태로 표시합니다.
                cap.release()  # VideoCapture 자원을 해제합니다.
                time.sleep(RECONNECT_DELAY)  # 재연결 전 잠시 대기합니다.
                continue  # 루프 처음으로 돌아갑니다.

            self.connected = True  # 연결 상태를 True로 설정합니다.
            self.status = "CONNECTED"  # 연결 상태 문구를 설정합니다.
            self._fc = 0  # FPS 카운터를 초기화합니다.
            self._t0 = time.time()  # FPS 계산 기준 시간을 초기화합니다.

            while self._running:  # 연결된 동안 계속 프레임을 읽습니다.
                ret, frame = cap.read()  # RTSP 영상에서 프레임 1장을 읽습니다.
                if not ret or frame is None:  # 프레임 수신 실패 여부를 확인합니다.
                    self.connected = False  # 연결 상태를 False로 설정합니다.
                    self.status = "NO SIGNAL"  # 신호 없음 상태로 표시합니다.
                    break  # 내부 루프를 빠져나가 재연결합니다.

                with self._lock:  # 프레임 저장 중 충돌을 막기 위해 잠급니다.
                    self._frame = frame  # 최신 프레임을 저장합니다.

                self._fc += 1  # FPS 계산용 프레임 수를 증가시킵니다.
                elapsed = time.time() - self._t0  # FPS 계산 기준 시간 이후 지난 시간을 계산합니다.
                if elapsed >= 0.5:  # 0.5초마다 FPS를 갱신합니다.
                    self.fps = self._fc / elapsed  # FPS를 계산합니다.
                    self._fc = 0  # 프레임 카운터를 초기화합니다.
                    self._t0 = time.time()  # FPS 계산 기준 시간을 갱신합니다.

            cap.release()  # 스트림 연결이 끊기면 자원을 해제합니다.
            if self._running:  # 프로그램이 종료 중이 아니면 재연결을 준비합니다.
                time.sleep(RECONNECT_DELAY)  # 재연결 전 잠시 대기합니다.

    def read(self):  # 최신 프레임을 복사해서 반환하는 함수입니다.
        with self._lock:  # 프레임 읽기 중 충돌을 막기 위해 잠급니다.
            return None if self._frame is None else self._frame.copy()  # 프레임이 있으면 복사본을 반환합니다.

    def stop(self):  # 카메라 수신 스레드를 종료하는 함수입니다.
        self._running = False  # 실행 상태를 False로 설정합니다.
        if self._thread is not None:  # 스레드가 존재하는지 확인합니다.
            self._thread.join(timeout=1.0)  # 최대 1초 동안 스레드 종료를 기다립니다.


def safe_crop(frame: np.ndarray) -> np.ndarray:  # ROI 영역을 안전하게 잘라내는 함수입니다.
    h, w = frame.shape[:2]  # 입력 영상의 높이와 가로 크기를 가져옵니다.
    x1 = max(0, min(ROI_X1, w - 1))  # ROI 왼쪽 좌표가 영상 범위를 벗어나지 않게 보정합니다.
    y1 = max(0, min(ROI_Y1, h - 1))  # ROI 위쪽 좌표가 영상 범위를 벗어나지 않게 보정합니다.
    x2 = max(1, min(ROI_X2, w))  # ROI 오른쪽 좌표가 영상 범위를 벗어나지 않게 보정합니다.
    y2 = max(1, min(ROI_Y2, h))  # ROI 아래쪽 좌표가 영상 범위를 벗어나지 않게 보정합니다.
    if x2 <= x1 or y2 <= y1:  # ROI 좌표가 잘못되어 영역이 없는지 확인합니다.
        return frame.copy()  # ROI가 잘못되면 전체 프레임을 반환합니다.
    return frame[y1:y2, x1:x2].copy()  # 지정한 ROI 영역만 잘라서 반환합니다.


def draw_grid(img: np.ndarray):  # 영상 위에 픽셀 좌표 격자를 그리는 함수입니다.
    h, w = img.shape[:2]  # 격자를 그릴 영상의 높이와 가로 크기를 가져옵니다.
    for x in range(GRID_SPACING, w, GRID_SPACING):  # 지정 간격마다 세로선 위치를 계산합니다.
        cv2.line(img, (x, 0), (x, h), GRID_COLOR, GRID_THICKNESS, cv2.LINE_AA)  # 세로 격자선을 그립니다.
        if GRID_LABEL:  # 좌표 숫자 표시가 켜져 있는지 확인합니다.
            cv2.putText(img, str(x), (x + 2, 14), FONT, 0.4, GRID_COLOR, 1, cv2.LINE_AA)  # 세로선의 X좌표를 위쪽에 표시합니다.
    for y in range(GRID_SPACING, h, GRID_SPACING):  # 지정 간격마다 가로선 위치를 계산합니다.
        cv2.line(img, (0, y), (w, y), GRID_COLOR, GRID_THICKNESS, cv2.LINE_AA)  # 가로 격자선을 그립니다.
        if GRID_LABEL:  # 좌표 숫자 표시가 켜져 있는지 확인합니다.
            cv2.putText(img, str(y), (2, y - 3), FONT, 0.4, GRID_COLOR, 1, cv2.LINE_AA)  # 가로선의 Y좌표를 왼쪽에 표시합니다.


def filter_long_horizontal(mask: np.ndarray, min_width: int) -> np.ndarray:  # 가로로 충분히 긴 엣지 덩어리만 남기는 함수입니다.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)  # 흰색 덩어리를 라벨링하고 각 덩어리 정보를 구합니다.
    out = np.zeros_like(mask)  # 결과를 담을 검은색 마스크를 만듭니다.
    for i in range(1, num):  # 배경(0번)을 제외한 각 덩어리를 확인합니다.
        if stats[i, cv2.CC_STAT_WIDTH] >= min_width:  # 덩어리의 가로 폭이 기준 이상인지 확인합니다.
            out[labels == i] = 255  # 기준을 만족하는 긴 가로 덩어리만 결과에 남깁니다.
    return out  # 긴 가로 엣지만 남은 마스크를 반환합니다.


def detect_horizontal_edges(blur: np.ndarray) -> np.ndarray:  # 완만한 가로 경계까지 잡아내는 가로 엣지 검출 함수입니다.
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL)  # 좌우(가로) 방향 밝기 변화를 계산합니다.
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL)  # 상하(세로) 방향 밝기 변화를 계산합니다. 가로 엣지일수록 값이 큽니다.
    abs_gx = np.abs(gx)  # 가로 방향 변화의 크기를 구합니다.
    abs_gy = np.abs(gy)  # 세로 방향 변화의 크기를 구합니다. 이 값 자체를 절대 기준으로 판정합니다.
    strong = abs_gy >= HORIZONTAL_GRAD_THRESHOLD  # 세로 변화가 충분히 강한 픽셀을 고릅니다. 완만한 경계도 잡히도록 낮게 잡습니다.
    dominant = abs_gy >= (HORIZONTAL_DOMINANCE * abs_gx)  # 세로 변화가 가로 변화보다 우세한(=가로 방향인) 픽셀을 고릅니다.
    mask = (strong & dominant).astype(np.uint8) * 255  # 두 조건을 모두 만족하는 가로 엣지만 흰색 마스크로 만듭니다.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZONTAL_MORPH_W, HORIZONTAL_MORPH_H))  # 가로로 긴 형태학 커널을 만듭니다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 가로 커널로 닫힘 연산을 해 끊긴 가로선을 이어 붙입니다.
    mask = filter_long_horizontal(mask, MIN_EDGE_WIDTH)  # 가로로 긴 경계선만 남기고 짧은 스크래치 잡음은 제거합니다.
    return mask  # 최종 가로 엣지 마스크를 반환합니다.


def clamp_measure_box(img: np.ndarray):  # 측정 박스 좌표가 영상 범위를 벗어나지 않도록 보정하는 함수입니다.
    h, w = img.shape[:2]  # 입력 영상의 높이와 가로 크기를 가져옵니다.
    x1 = max(0, min(MEASURE_X1, w - 1))  # 측정 박스 왼쪽 좌표를 영상 내부로 제한합니다.
    y1 = max(0, min(MEASURE_Y1, h - 1))  # 측정 박스 위쪽 좌표를 영상 내부로 제한합니다.
    x2 = max(1, min(MEASURE_X2, w))  # 측정 박스 오른쪽 좌표를 영상 내부로 제한합니다.
    y2 = max(1, min(MEASURE_Y2, h))  # 측정 박스 아래쪽 좌표를 영상 내부로 제한합니다.
    if x2 <= x1:  # 오른쪽 좌표가 왼쪽보다 작거나 같으면 박스 폭이 없는 상태입니다.
        x2 = min(w, x1 + 1)  # 최소 1픽셀 폭이 생기도록 보정합니다.
    if y2 <= y1:  # 아래쪽 좌표가 위쪽보다 작거나 같으면 박스 높이가 없는 상태입니다.
        y2 = min(h, y1 + 1)  # 최소 1픽셀 높이가 생기도록 보정합니다.
    return x1, y1, x2, y2  # 보정된 측정 박스 좌표를 반환합니다.


def cluster_rows(rows: np.ndarray):  # 경계 후보 행들을 가까운 행끼리 묶는 함수입니다.
    if rows.size == 0:  # 후보 행이 하나도 없으면 빈 리스트를 반환합니다.
        return []  # 검출 실패 상태를 의미합니다.
    clusters = []  # 묶인 행 그룹들을 저장할 리스트입니다.
    start = int(rows[0])  # 현재 그룹의 시작 행입니다.
    prev = int(rows[0])  # 현재 그룹에서 마지막으로 확인한 행입니다.
    for r in rows[1:]:  # 두 번째 후보 행부터 순서대로 확인합니다.
        r = int(r)  # 넘파이 정수를 파이썬 정수로 변환합니다.
        if r - prev <= ROW_CLUSTER_GAP:  # 이전 행과 충분히 가까우면 같은 경계로 판단합니다.
            prev = r  # 현재 그룹의 마지막 행을 갱신합니다.
        else:  # 이전 행과 너무 멀면 새로운 경계로 판단합니다.
            clusters.append((start, prev))  # 완성된 경계 그룹을 저장합니다.
            start = r  # 새 그룹의 시작 행을 현재 행으로 설정합니다.
            prev = r  # 새 그룹의 마지막 행을 현재 행으로 설정합니다.
    clusters.append((start, prev))  # 마지막 경계 그룹을 저장합니다.
    return clusters  # 경계 그룹 리스트를 반환합니다.


def measure_two_horizontal_edges(horiz: np.ndarray):  # 측정 박스 안에서 위/아래 가로 엣지 2개 사이의 세로 거리를 계산하는 함수입니다.
    x1, y1, x2, y2 = clamp_measure_box(horiz)  # 확대 ROI 기준 측정 박스 좌표를 안전하게 가져옵니다.
    roi_mask = horiz[y1:y2, x1:x2]  # 가로 엣지 마스크에서 측정 박스 영역만 잘라냅니다.
    row_counts = np.count_nonzero(roi_mask > 0, axis=1)  # 각 행마다 초록 엣지 픽셀이 몇 개 있는지 계산합니다.
    candidate_rows = np.where(row_counts >= ROW_MIN_PIXELS)[0]  # 엣지 픽셀이 충분한 행만 경계 후보로 선택합니다.
    clusters = cluster_rows(candidate_rows)  # 가까운 후보 행들을 하나의 경계 그룹으로 묶습니다.
    if len(clusters) < 2:  # 위/아래 경계 2개를 찾지 못한 경우입니다.
        return None, (x1, y1, x2, y2), clusters  # 거리 없음 상태와 박스 좌표를 반환합니다.
    centers = [int(round((a + b) / 2)) for a, b in clusters]  # 각 경계 그룹의 중심 Y좌표를 계산합니다.
    strengths = []  # 각 경계 그룹의 강도를 저장할 리스트입니다.
    for a, b in clusters:  # 각 경계 그룹을 순회합니다.
        strengths.append(int(row_counts[a:b + 1].sum()))  # 그룹 내부 초록 픽셀 총합을 강도로 사용합니다.
    order = np.argsort(strengths)[::-1]  # 강도가 큰 경계부터 정렬합니다.
    top_two = sorted([centers[int(order[0])], centers[int(order[1])]])  # 가장 강한 경계 2개의 Y중심을 위/아래 순서로 정렬합니다.
    y_top = y1 + top_two[0]  # 확대 ROI 전체 좌표계 기준 위쪽 경계 Y좌표입니다.
    y_bottom = y1 + top_two[1]  # 확대 ROI 전체 좌표계 기준 아래쪽 경계 Y좌표입니다.
    distance_px = abs(y_bottom - y_top)  # 두 경계 사이의 세로 픽셀 거리를 계산합니다.
    result = {"top_y": y_top, "bottom_y": y_bottom, "distance_px": distance_px}  # 측정 결과를 딕셔너리로 구성합니다.
    return result, (x1, y1, x2, y2), clusters  # 측정 결과와 박스 좌표를 반환합니다.


def _line_pop(img, p1, p2, color, thick=2):  # 검은 밑선을 깔아 어떤 배경에서도 잘 보이는 선을 그리는 보조 함수입니다.
    cv2.line(img, p1, p2, (0, 0, 0), thick + 3, cv2.LINE_AA)  # 먼저 두꺼운 검정 선을 깔아 외곽선을 만듭니다.
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)  # 그 위에 실제 색상 선을 그립니다.


def _arrow_pop(img, p1, p2, color, thick=2, tip=0.06):  # 검은 밑선을 깔아 잘 보이는 화살표를 그리는 보조 함수입니다.
    cv2.arrowedLine(img, p1, p2, (0, 0, 0), thick + 3, cv2.LINE_AA, tipLength=tip)  # 먼저 두꺼운 검정 화살표로 외곽선을 만듭니다.
    cv2.arrowedLine(img, p1, p2, color, thick, cv2.LINE_AA, tipLength=tip)  # 그 위에 실제 색상 화살표를 그립니다.


def _label_box(img, text, org, text_color, scale=0.6):  # 반투명 검정 배경을 깔아 글자가 또렷하게 보이도록 하는 보조 함수입니다.
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 2)  # 글자 크기를 측정합니다.
    x, y = org  # 글자 기준 좌표를 풀어냅니다.
    h, w = img.shape[:2]  # 영상 크기를 가져와 배경 상자가 화면을 벗어나지 않게 씁니다.
    x = max(2, min(x, w - tw - 6))  # 배경 상자의 X를 화면 안으로 보정합니다.
    y = max(th + 6, min(y, h - 4))  # 배경 상자의 Y를 화면 안으로 보정합니다.
    p1 = (x - 4, y - th - 6)  # 배경 상자의 왼쪽 위 좌표입니다.
    p2 = (x + tw + 4, y + base)  # 배경 상자의 오른쪽 아래 좌표입니다.
    sub = img[max(0, p1[1]):max(0, p2[1]), max(0, p1[0]):max(0, p2[0])]  # 배경 상자가 덮을 영역을 잘라냅니다.
    if sub.size > 0:  # 잘라낸 영역이 유효한지 확인합니다.
        dark = (sub.astype(np.float32) * 0.35).astype(np.uint8)  # 그 영역을 어둡게 만들어 반투명 검정 배경 효과를 줍니다.
        img[max(0, p1[1]):max(0, p2[1]), max(0, p1[0]):max(0, p2[0])] = dark  # 어둡게 만든 영역을 다시 넣습니다.
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)  # 글자 외곽선을 검정으로 먼저 그립니다.
    cv2.putText(img, text, (x, y), FONT, scale, text_color, 1, cv2.LINE_AA)  # 실제 글자를 지정 색으로 그립니다.
    return (x, y)  # 실제 사용된 글자 좌표를 반환합니다.


def draw_distance_overlay(img: np.ndarray, result, box):  # 확대 실영상/엣지 영상 위에 측정 박스와 거리값을 시각화하는 함수입니다.
    x1, y1, x2, y2 = box  # 측정 박스 좌표를 풀어서 사용합니다.
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)  # 박스 외곽선을 검정으로 먼저 그려 어떤 배경에서도 보이게 합니다.
    cv2.rectangle(img, (x1, y1), (x2, y2), MEASURE_BOX_COLOR, 2)  # 측정 대상 영역을 빨간 박스로 표시합니다.

    if result is None:  # 거리 측정에 실패한 경우입니다.
        _label_box(img, "Distance: NOT FOUND", (x1 + 4, max(24, y1 - 8)), MEASURE_TEXT_COLOR)  # 실패 문구를 배경과 함께 표시합니다.
        return img  # 시각화가 적용된 이미지를 반환합니다.

    y_top = int(result["top_y"])  # 위쪽 경계 Y좌표를 가져옵니다.
    y_bottom = int(result["bottom_y"])  # 아래쪽 경계 Y좌표를 가져옵니다.
    distance_px = int(result["distance_px"])  # 확대 영상 기준 픽셀 거리값을 가져옵니다.
    mid_x = int((x1 + x2) / 2)  # 화살표를 그릴 X중심 좌표입니다.

    overlay = img.copy()  # 반투명 채움을 위해 원본 복사본을 만듭니다.
    cv2.rectangle(overlay, (x1, y_top), (x2, y_bottom), MEASURE_FILL_COLOR, -1)  # 두 경계 사이 영역을 채워 측정 구간을 강조합니다.
    cv2.addWeighted(overlay, MEASURE_FILL_ALPHA, img, 1 - MEASURE_FILL_ALPHA, 0, img)  # 채운 영역을 반투명하게 합성합니다.

    _line_pop(img, (x1, y_top), (x2, y_top), MEASURE_LINE_COLOR, 2)  # 위쪽 경계 기준선을 잘 보이게 그립니다.
    _line_pop(img, (x1, y_bottom), (x2, y_bottom), MEASURE_LINE_COLOR, 2)  # 아래쪽 경계 기준선을 잘 보이게 그립니다.

    _arrow_pop(img, (mid_x, y_top), (mid_x, y_bottom), MEASURE_ARROW_COLOR, 2)  # 위에서 아래로 거리 화살표를 그립니다.
    _arrow_pop(img, (mid_x, y_bottom), (mid_x, y_top), MEASURE_ARROW_COLOR, 2)  # 아래에서 위로 거리 화살표를 그려 양방향 치수선으로 만듭니다.

    for py in (y_top, y_bottom):  # 두 경계선의 중앙 지점마다 점을 찍습니다.
        cv2.circle(img, (mid_x, py), 4, (255, 255, 255), -1, cv2.LINE_AA)  # 흰색 점을 찍습니다.
        cv2.circle(img, (mid_x, py), 4, (0, 0, 0), 1, cv2.LINE_AA)  # 흰색 점 둘레를 검정으로 감쌉니다.

    distance_orig = distance_px / ZOOM_SCALE  # 확대 배율을 나눠 원본(확대 전) 픽셀 거리로 환산합니다.
    if PIXEL_TO_MM is None:  # mm 환산값이 설정되지 않은 경우입니다.
        label = f"{distance_px}px (orig {distance_orig:.1f}px)"  # 확대/원본 픽셀 거리를 표시하는 문구를 만듭니다.
    else:  # mm 환산값이 설정된 경우입니다.
        label = f"{distance_px}px = {distance_px * PIXEL_TO_MM:.3f}mm"  # 확대 픽셀 거리와 mm를 함께 표시하는 문구를 만듭니다.
    _label_box(img, label, (mid_x + 12, (y_top + y_bottom) // 2 + 6), MEASURE_TEXT_COLOR)  # 화살표 옆에 거리값을 배경과 함께 표시합니다.
    return img  # 시각화가 적용된 이미지를 반환합니다.


def make_roi_zoom_canny(frame: np.ndarray):  # 원본, ROI 확대 영상, 가로 엣지 영상, 거리 측정 결과와 박스를 만드는 함수입니다.
    original = cv2.resize(frame, (VIEW_W, VIEW_H))  # 원본 영상을 표시 크기로 리사이즈합니다.
    roi = safe_crop(original)  # 원본 표시 영상에서 검사할 ROI만 잘라냅니다.
    zoom = cv2.resize(roi, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_CUBIC)  # ROI를 지정 배율로 확대합니다.
    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)  # 엣지 처리를 위해 확대 영상을 흑백으로 변환합니다.
    blur = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)  # 표면 스크래치 잡음을 줄이기 위해 블러를 적용합니다.
    canny = cv2.Canny(blur, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)  # 참고용 전체 엣지를 검출합니다.
    horiz = detect_horizontal_edges(blur)  # 완만한 경계까지 포함해 가로 엣지를 검출합니다.
    edge_bgr = np.zeros((zoom.shape[0], zoom.shape[1], 3), dtype=np.uint8)  # 결과를 그릴 검은색 3채널 캔버스를 만듭니다.
    edge_bgr[canny > 0] = (70, 70, 70)  # 참고용 전체 엣지는 흐린 회색으로 표시합니다.
    edge_bgr[horiz > 0] = (0, 255, 0)  # 검출된 가로 엣지는 밝은 초록색으로 강조 표시합니다.

    distance_result = None  # 거리 측정 결과를 저장할 변수를 초기화합니다.
    measure_box = None  # 측정 박스 좌표를 저장할 변수를 초기화합니다.
    if MEASURE_ENABLED:  # 거리 측정 기능이 켜져 있는지 확인합니다.
        distance_result, measure_box, _ = measure_two_horizontal_edges(horiz)  # 측정 박스 안에서 위/아래 가로 엣지 사이 거리를 계산합니다.

    return original, zoom, edge_bgr, distance_result, measure_box  # 원본, 확대 ROI, 가로 엣지 영상, 거리 결과, 박스를 반환합니다.


def draw_original_roi_box(original: np.ndarray):  # 원본 화면에 ROI 위치 박스를 그리는 함수입니다.
    cv2.rectangle(original, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)  # 원본 화면에 노란색 ROI 박스를 표시합니다.


def draw_status(img: np.ndarray, text: str, y: int):  # 영상 위에 상태 글자를 표시하는 함수입니다.
    cv2.putText(img, text, (10, y), FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)  # 글자 외곽선을 검정색으로 먼저 그립니다.
    cv2.putText(img, text, (10, y), FONT, 0.7, (0, 255, 0), 1, cv2.LINE_AA)  # 실제 글자를 초록색으로 그립니다.


def fit_height(img: np.ndarray, target_h: int) -> np.ndarray:  # 영상 높이를 기준에 맞춰 리사이즈하는 함수입니다.
    h, w = img.shape[:2]  # 입력 영상의 높이와 가로 크기를 가져옵니다.
    scale = target_h / h  # 목표 높이에 맞추기 위한 배율을 계산합니다.
    new_w = max(1, int(w * scale))  # 배율에 맞춰 새 가로 크기를 계산합니다.
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)  # 목표 높이에 맞춰 영상을 리사이즈합니다.


def compose_frame(frame, grid_on):  # 한 프레임을 받아 측정 시각화까지 끝난 원본/확대/엣지 영상을 만드는 함수입니다.
    if frame is None:  # 아직 프레임이 수신되지 않았는지 확인합니다.
        original = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)  # 검은색 원본 대기 화면을 만듭니다.
        zoom = np.zeros((int((ROI_Y2 - ROI_Y1) * ZOOM_SCALE), int((ROI_X2 - ROI_X1) * ZOOM_SCALE), 3), dtype=np.uint8)  # 검은색 확대 ROI 대기 화면을 만듭니다.
        edge = zoom.copy()  # 검은색 가로 엣지 대기 화면을 만듭니다.
        draw_status(zoom, "Waiting for stream...", 30)  # 확대 화면에 대기 문구를 표시합니다.
        draw_status(edge, "Waiting for stream...", 30)  # 엣지 화면에 대기 문구를 표시합니다.
        return original, zoom, edge, None  # 대기 화면과 빈 측정 결과를 반환합니다.

    original, zoom, edge, distance_result, measure_box = make_roi_zoom_canny(frame)  # 원본/확대/엣지 영상과 측정 결과를 생성합니다.
    draw_original_roi_box(original)  # 원본 화면에 ROI 박스를 표시합니다.

    if grid_on:  # 격자 표시가 켜져 있는지 확인합니다.
        draw_grid(original)  # 원본 화면에 격자를 그립니다.
        draw_grid(zoom)  # 확대 ROI 화면에 격자를 그립니다.
        draw_grid(edge)  # 가로 엣지 화면에 격자를 그립니다.

    if MEASURE_ENABLED and measure_box is not None:  # 측정 기능이 켜져 있고 박스가 있는지 확인합니다.
        draw_distance_overlay(zoom, distance_result, measure_box)  # 3배 확대 실영상에 측정 시각화를 그립니다.
        draw_distance_overlay(edge, distance_result, measure_box)  # 가로 엣지 영상에도 동일한 측정 시각화를 그립니다.

    return original, zoom, edge, distance_result  # 시각화까지 끝난 세 영상과 측정 결과를 반환합니다.


def main():  # 프로그램 메인 함수입니다.
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)  # 스냅샷 저장 폴더가 없으면 생성합니다.

    grid_on = GRID_ENABLED  # 실행 중 격자 표시 여부를 저장하는 변수입니다.

    cam = CameraStream(CAMERA_NAME, CAMERA_URL)  # Parking RTSP 카메라 객체를 생성합니다.
    cam.start()  # 카메라 수신 스레드를 시작합니다.

    plc = PlcModbusWriter()  # PLC D영역 쓰기용 Modbus RTU 객체를 생성합니다.
    plc.connect()  # 프로그램 시작 시 PLC 연결을 한 번 시도합니다.

    plc.write_d(1, force=True)
    time.sleep(0.5)
    plc.write_d(0, force=True)


    distance_state = DistanceNotFoundState()  # Distance NOT FOUND 상태 안정 판정 객체를 생성합니다.

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 크기 조절 가능한 OpenCV 창을 생성합니다.

    print("Parking RTSP ROI 확대 + 가로 엣지 검출 + 거리 측정 시작")  # 프로그램 시작 안내를 출력합니다.
    print(f"카메라 주소: {CAMERA_URL}")  # 사용 중인 RTSP 주소를 출력합니다.
    print(f"ROI: x={ROI_X1}~{ROI_X2}, y={ROI_Y1}~{ROI_Y2}")  # 현재 ROI 좌표를 출력합니다.
    print(f"확대 배율: {ZOOM_SCALE}배")  # 현재 확대 배율을 출력합니다.
    print(f"가로 엣지 임계: {HORIZONTAL_GRAD_THRESHOLD}, 최소 길이: {MIN_EDGE_WIDTH}px")  # 현재 가로 엣지 검출 설정을 출력합니다.
    print(f"측정 박스: x={MEASURE_X1}~{MEASURE_X2}, y={MEASURE_Y1}~{MEASURE_Y2} (박스 위치는 MEASURE_X1~Y2 값으로 조정)")  # 현재 거리 측정 박스 좌표를 출력합니다.
    print("조작키: [q] 종료, [s] 스냅샷 저장, [g] 격자 표시 전환")  # 조작키를 안내합니다.
    print(f"PLC: {PLC_PORT}, {PLC_BAUDRATE},{PLC_BYTESIZE},{PLC_PARITY},{PLC_STOPBITS}, slave={PLC_SLAVE_ID}, D주소={PLC_D_REGISTER_ADDRESS}")  # PLC 통신 설정을 출력합니다.

    try:  # 종료 시 자원 해제를 보장하기 위해 try 블록을 사용합니다.
        while True:  # 사용자가 종료할 때까지 반복합니다.
            frame = cam.read()  # 카메라에서 최신 프레임을 읽습니다.

            original, zoom, edge, distance_result = compose_frame(frame, grid_on)  # 측정 시각화까지 끝난 영상들을 생성합니다.

            plc_value = distance_state.update(distance_result)  # 현재 측정 결과로 PLC에 새로 보낼 값이 있는지 판단합니다.
            if plc_value is not None:  # 새로 보낼 PLC 값이 있을 때만 통신합니다.
                plc.write_d(plc_value)  # RS-485 Modbus RTU로 PLC D영역에 상태값을 씁니다.

            draw_status(original, f"Original | {cam.status} | {cam.fps:.1f} fps", 30)  # 원본 화면 상태를 표시합니다.
            draw_status(original, "Yellow Box = ROI", 60)  # 원본 화면에 ROI 설명을 표시합니다.
            draw_status(zoom, f"ROI Zoom x{ZOOM_SCALE}", 30)  # 확대 ROI 화면 제목을 표시합니다.
            draw_status(edge, f"Horizontal Edge | thr={HORIZONTAL_GRAD_THRESHOLD}, min_w={MIN_EDGE_WIDTH}", 30)  # 가로 엣지 화면 제목을 표시합니다.
            draw_status(edge, "Green = Horizontal Edge / Gray = All", 60)  # 색상 범례를 표시합니다.
            draw_status(edge, f"PLC D{PLC_D_REGISTER_ADDRESS}: state={distance_state.current_state} port={PLC_PORT}", 120)  # PLC 전송 상태를 화면에 표시합니다.
            if distance_result is not None:  # 거리 측정값이 있는지 확인합니다.
                gap_px = distance_result['distance_px']  # 측정된 확대 픽셀 거리를 꺼냅니다.
                draw_status(edge, f"Measured gap = {gap_px} px (orig {gap_px / ZOOM_SCALE:.1f} px)", 90)  # 측정 거리값을 상태 문구로 한 번 더 표시합니다.

            original_fit = fit_height(original, 360)  # 원본 화면 높이를 360으로 맞춥니다.
            zoom_fit = fit_height(zoom, 360)  # 확대 ROI 화면 높이를 360으로 맞춥니다.
            edge_fit = fit_height(edge, 360)  # 가로 엣지 화면 높이를 360으로 맞춥니다.

            top_row = np.hstack((original_fit, zoom_fit))  # 위쪽 줄에 원본과 확대 ROI를 좌우로 붙입니다.
            bottom_row = edge_fit  # 아래쪽 줄은 가로 엣지 영상만 사용합니다.

            top_w = top_row.shape[1]  # 위쪽 줄의 가로 크기를 가져옵니다.
            bottom_w = bottom_row.shape[1]  # 아래쪽 줄의 가로 크기를 가져옵니다.

            if bottom_w < top_w:  # 엣지 영상이 위쪽 줄보다 좁은지 확인합니다.
                pad = np.zeros((bottom_row.shape[0], top_w - bottom_w, 3), dtype=np.uint8)  # 오른쪽에 붙일 검은 여백을 만듭니다.
                bottom_row = np.hstack((bottom_row, pad))  # 엣지 영상 오른쪽에 여백을 붙입니다.
            elif bottom_w > top_w:  # 엣지 영상이 위쪽 줄보다 넓은지 확인합니다.
                pad = np.zeros((top_row.shape[0], bottom_w - top_w, 3), dtype=np.uint8)  # 오른쪽에 붙일 검은 여백을 만듭니다.
                top_row = np.hstack((top_row, pad))  # 위쪽 줄 오른쪽에 여백을 붙입니다.

            display = np.vstack((top_row, bottom_row))  # 위쪽 줄과 아래쪽 줄을 세로로 붙입니다.

            cv2.imshow(WINDOW_NAME, display)  # 최종 화면을 표시합니다.

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:  # 창 X 버튼으로 닫혔는지 확인합니다.
                break  # 창이 닫혔으면 반복을 종료합니다.

            key = cv2.waitKey(20) & 0xFF  # 키 입력을 20ms 동안 기다립니다.

            if key == ord("q"):  # q 키가 눌렸는지 확인합니다.
                break  # q 키가 눌리면 종료합니다.

            elif key == ord("g"):  # g 키가 눌렸는지 확인합니다.
                grid_on = not grid_on  # 격자 표시 상태를 반전시킵니다.
                print(f"격자 표시: {'ON' if grid_on else 'OFF'}")  # 현재 격자 표시 상태를 출력합니다.

            elif key == ord("s") and frame is not None:  # s 키가 눌렸고 프레임이 있을 때 스냅샷을 저장합니다.
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 파일명용 문자열로 만듭니다.
                save_original, save_zoom, save_edge, _ = compose_frame(frame, grid_on)  # 측정 시각화까지 포함한 저장용 영상을 생성합니다.
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"Parking_original_roi_box_{ts}.jpg"), save_original)  # ROI 박스 포함 원본을 저장합니다.
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"Parking_roi_zoom_{ts}.jpg"), save_zoom)  # 측정 시각화가 포함된 확대 ROI 영상을 저장합니다.
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"Parking_roi_hedge_{ts}.jpg"), save_edge)  # 측정 시각화가 포함된 가로 엣지 영상을 저장합니다.
                print(f"스냅샷 저장 완료: ./{SNAPSHOT_DIR}/")  # 저장 완료 메시지를 출력합니다.

    finally:  # 프로그램 종료 시 항상 실행됩니다.
        plc.close()  # PLC RS-485 통신 포트를 닫습니다.
        cam.stop()  # 카메라 수신 스레드를 종료합니다.
        cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.
        print("프로그램을 종료합니다.")  # 종료 메시지를 출력합니다.


if __name__ == "__main__":  # 이 파일을 직접 실행할 때만 main을 실행합니다.
    main()  # 메인 함수를 실행합니다.
