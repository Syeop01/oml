**작동 방법**

터미널에서 파이썬 스크립트(.py)를 직접 실행할 경우 주피터랩 위젯(카메라 화면)이 출력되지 않으므로, 화면 모니터링이 가능한 아래의 주피터 노트북(.ipynb) 활용 방식을 권장함.

[1단계] TensorRT 모델 변환 (필수)
젯봇의 연산 지연을 막기 위해 원본 모델(.pth)을 고속 엔진으로 변환하는 과정을 우선 수행함.

젯봇 터미널을 열고 python3 convert_torch2trt.py 명령어를 실행함.

약 2~5분 소요 후, 폴더 내에 best_model_xy_trt.pth 엔진 파일이 생성됨을 확인함.

[2단계] 실행용 노트북(.ipynb) 파일 생성 및 코드 복사
주피터랩 파일 탐색기에서 모델 파일이 위치한 경로와 동일한 폴더에 새로운 .ipynb 파일을 생성함.

본 레포지토리의 lf_live_demo_trt_yahboom.py 파일을 더블클릭하여 연 뒤, 전체 코드를 복사(Ctrl+A, Ctrl+C)함.

새로 만든 .ipynb 파일의 빈 셀에 복사한 전체 코드를 붙여넣기(Ctrl+V) 함.

[3단계] 자율주행 실행 및 실시간 모니터링
젯봇을 트랙 위에 올리고 메인 전원(배터리 스위치)을 켬.

코드가 입력된 주피터랩 셀을 실행(Shift + Enter)함.

"Loading TRT Engine..." 메시지 출력 후 약 10~30초를 대기함.

로딩 완료 시 젯봇이 주행을 시작하며, 주피터랩 브라우저 내 위젯을 통해 15프레임 주기로 갱신되는 로봇의 시야(카메라)를 실시간으로 확인함.

주행을 중단할 때는 셀 실행을 멈추기 위해 주피터랩 상단의 정지(■) 버튼을 클릭하여 안전하게 종료함.
___________________________________________________________________________________________________________________________________________
**파라미터 튜닝**

주행 환경(트랙 곡률, 배터리 등)에 따라 동작이 달라질 경우, 복사해 넣은 .ipynb 셀 내부의 파라미터를 직접 수정하며 최적화함.

speed_gain_value: 직진 기본 속도. (바퀴가 돌지 않을 경우 0.40 등 마찰력을 이길 때까지 상향함)

steering_gain_value: 회전 민감도. (차선을 자주 이탈하면 상향, 지그재그로 떨면 하향함)

steering_dgain_value: 코너링 안정화(오버슈트 방지)를 위한 미분항으로 0.05 근처에서 미세 조정함.

___________________________________________________________________________________________________________________________________________
**Usage Instructions**

Executing the Python script (.py) directly in the terminal will not display the JupyterLab widget (camera view). Therefore, the following Jupyter Notebook (.ipynb) method is recommended to enable visual monitoring.

[Step 1] TensorRT Model Conversion (Required)
Convert the original model (.pth) into a high-speed engine first to prevent computational latency on the Jetbot.

Open the Jetbot terminal and execute the python3 convert_torch2trt.py command.

After approximately 2–5 minutes, confirm that the best_model_xy_trt.pth engine file has been generated in the folder.

[Step 2] Create an Executable Notebook (.ipynb) File and Copy Code
Create a new .ipynb file in the same folder where the model file is located using the JupyterLab file explorer.

Double-click to open the lf_live_demo_trt_yahboom.py file from this repository, and copy the entire code (Ctrl+A, Ctrl+C).

Paste the copied code (Ctrl+V) into an empty cell of the newly created .ipynb file.

[Step 3] Execute Autonomous Driving and Real-Time Monitoring
Place the Jetbot on the track and turn on the main power (battery switch).

Execute the JupyterLab cell containing the code (Shift + Enter).

Wait for about 10–30 seconds after the "TRT 엔진 로딩 중..." (Loading TRT Engine...) message is displayed.

Upon completion of the loading process, the Jetbot will start driving. Monitor the robot's vision (camera) in real-time through the JupyterLab browser widget, which updates every 15 frames.

To stop driving, safely terminate the process by clicking the Stop (■) button at the top of JupyterLab to halt cell execution.

**Parameter Tuning**

If the driving behavior varies depending on the environment (track curvature, battery level, etc.), optimize it by directly modifying the parameters inside the pasted .ipynb cell.

speed_gain_value: Base straight speed. (If the wheels do not turn, increase it to 0.40 or until it overcomes static friction).

steering_gain_value: Steering sensitivity. (Increase if the robot frequently leaves the lane; decrease if it wobbles in a zigzag pattern).

steering_dgain_value: Derivative term for cornering stability (preventing overshoot). Fine-tune it around the value of 0.05.
