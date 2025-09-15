<!-- AirDraw — README (Ukr) -->
<section style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; line-height:1.5; color:#111;">
  <h1 style="margin-bottom:0.1em;">🎨 AirDraw</h1>
  <p style="margin-top:0.2em; color:#444;">
    Програма на Python, яка дозволяє малювати у повітрі за допомогою жестів рук, що розпізнаються веб-камерою.
    Права рука відповідає за малювання та зміну кольору, ліва рука — за регулювання товщини пензля.
  </p>
<p>
Цей проєкт базується на <a href="https://github.com/pratham-bhatnagar/Gesture-Volume-Control.git" target="_blank">Gesture Volume Control</a>, який був адаптований під малювання замість регулювання гучності.
</p>
  <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">

  <h2 style="font-size:1.15rem;margin-bottom:6px;">Функціонал</h2>
  <ul style="margin-top:0;color:#333">
    <li>Малювання ліній в режимі реального часу за допомогою вказівного пальця правої руки.</li>
    <li>Вибір кольору жестами правої руки:
      <ul>
        <li>1 палець (вказівний) → червоний</li>
        <li>2 пальці (вказівний + середній) → синій</li>
        <li>3 пальці (вказівний + середній + безіменний) → зелений</li>
      </ul>
    </li>
    <li>Контроль товщини: ліва рука — відстань між великим і вказівним пальцем визначає товщину пензля; діаметр відображається кружком на екрані.</li>
    <li>Віртуальне полотно (canvas) накладається на відеопотік з камери.</li>
  </ul>

  <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">

  <h2 style="font-size:1.15rem;margin-bottom:6px;">Вимоги</h2>
  <ul style="margin-top:0;color:#333">
    <li>Python 3.12.10 (рекомендується)</li>
    <li>Бібліотеки: <code>opencv-python</code>, <code>mediapipe</code>, <code>numpy</code></li>
  </ul>

  <h3 style="margin-top:8px;font-size:1rem;">Встановлення залежностей</h3>
  <pre style="background:#f6f8fa;border:1px solid #e1e4e8;padding:12px;border-radius:6px;overflow:auto;">
<code>pip install opencv-python mediapipe numpy</code>
  </pre>

  <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">

  <h2 style="font-size:1.15rem;margin-bottom:6px;">Запуск</h2>
  <p style="margin-top:0;color:#333">Запусти основний файл проєкту (наприклад <code>Lab1.py</code>):</p>
  <pre style="background:#f6f8fa;border:1px solid #e1e4e8;padding:12px;border-radius:6px;overflow:auto;">
<code>python Lab1.py</code>
  </pre>

  <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">

  <h2 style="font-size:1.15rem;margin-bottom:6px;">Короткий опис роботи</h2>
  <p style="margin-top:0;color:#333">
    Кожен кадр з камери обробляється через Mediapipe — визначаються координати ключових точок рук.
    Правою рукою користувач малює: кінчик вказівного пальця задає траєкторію; підняті пальці визначають колір.
    Ліва рука регулює товщину — відстань між великим та вказівним пальцем перетворюється в розмір пензля, який також візуалізується колом.
    Малюнок зберігається у вигляді шару (canvas) і накладається на відео.
  </p>
<section style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; line-height:1.5; color:#111;">
  <h2>📝 Пояснення коду</h2>

  <b>Імпорт бібліотек</b>
  <pre><code class="language-python">import cv2
import mediapipe as mp
import numpy as np
import math
</code></pre>
  <p>Підключаємо <b>OpenCV</b> для роботи з відео, <b>MediaPipe</b> для розпізнавання рук, <b>NumPy</b> для математики та <b>math</b> для обчислення відстаней.</p>
  <hr>

  <b>Ініціалізація моделі рук</b>
  <pre><code class="language-python">mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
</code></pre>
  <p>Використовуємо MediaPipe Hands для виявлення максимум двох рук з порогом впевненості 0.7.</p>
  <hr>

  <b>Налаштування камери</b>
  <pre><code class="language-python">cap = cv2.VideoCapture(0)
</code></pre>
  <p>Відкриваємо вебкамеру (0 = стандартна камера пристрою).</p>
  <hr>

  <b>Змінні для малювання</b>
  <pre><code class="language-python">canvas = None
brush_thickness = 5
prev_x, prev_y = None, None
brush_color = (0, 0, 255)
</code></pre>
  <p>Створюємо "полотно", задаємо товщину пензля, попередні координати та початковий колір (червоний).</p>
  <hr>

  <b>Функція визначення піднятих пальців</b>
  <p> Хоча функція відображення точок на долонях була видалена для чистоти інтерфейсу, ці ключові точки все одно активно використовуються в коді.<br>
</p>
<div align="center">
    <img alt="mediapipeLogo" src="hand_landmarks_docs.png" height="200 x    " />
</div>
  <pre><code class="language-python">def fingers_up(hand_landmarks, h):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    for tip in tips:
        y_tip = int(hand_landmarks.landmark[tip].y * h)
        y_base = int(hand_landmarks.landmark[tip - 2].y * h)
        fingers.append(1 if y_tip < y_base else 0)
    return fingers
</code></pre>
  <p>Порівнює положення кінчиків пальців із суглобами. Якщо кінчик вище — палець піднятий (1), інакше — опущений (0).</p>
  <hr>

  <b>Основний цикл</b>
  <pre><code class="language-python">while True:
    ret, frame = cap.read()
    if not ret:
        break
</code></pre>
  <p>Читаємо кадри з камери. Якщо камера не працює — виходимо з циклу.</p>
  <hr>

  <b>Підготовка кадру</b>
  <pre><code class="language-python">frame = cv2.flip(frame, 1)
h, w, _ = frame.shape

if canvas is None:
    canvas = np.zeros_like(frame)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb)
</code></pre>
  <p>Дзеркалимо картинку, створюємо "полотно" того ж розміру, переводимо в RGB та передаємо в MediaPipe для обробки.</p>
  <hr>

  <b>Розпізнавання рук</b>
  <pre><code class="language-python">if results.multi_hand_landmarks and results.multi_handedness:
    hand_info = list(zip(results.multi_handedness, results.multi_hand_landmarks))
</code></pre>
  <p>Якщо знайдені руки — беремо їхні ключові точки та інформацію, яка рука (ліва чи права).</p>
  <hr>
<b>Перебираємо всі знайдені руки у кадрі.</b>
<pre><code class="language-python">
for hand_type, hand_landmarks in hand_info:
    label = hand_type.classification[0].label
    x_index = int(hand_landmarks.landmark[8].x * w)
    y_index = int(hand_landmarks.landmark[8].y * h)
    x_thumb = int(hand_landmarks.landmark[4].x * w)
    y_thumb = int(hand_landmarks.landmark[4].y * h)
</code></pre>
<p>
<code>hand_type</code> — інформація про руку (ліва/права).<br> 
<code>hand_landmarks</code> — координати ключових точок.<br>
<code>label</code> визначає, яка це рука: ліва чи права.<br>
<code>x_index</code> і <code>y_index</code> — координати вказівного пальця для відображення та малювання.<br>
<code>x_thumb</code> і <code>y_thumb</code> — координати великого пальця, використовуються для обчислення відстані між пальцями (для товщини лінії).
</p>
<hr>
  <b>Обробка правої руки (малювання)</b>
  <pre><code class="language-python">if label == "Right":
    fingers = fingers_up(hand_landmarks, h)

    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
        brush_color = (0, 0, 255)  # Червоний
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
        brush_color = (255, 0, 0)  # Синій
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
        brush_color = (0, 255, 0)  # Зелений

    if prev_x is None or prev_y is None:
        prev_x, prev_y = x_index, y_index

    cv2.line(canvas, (prev_x, prev_y), (x_index, y_index), brush_color, brush_thickness)
    prev_x, prev_y = x_index, y_index
</code></pre>
  <p>Жести правої руки визначають колір пензля: 1 палець — червоний, 2 — синій, 3 — зелений. Вказівний малює лінію на "полотні".</p>
  <hr>

  <b>Обробка лівої руки (товщина)</b>
  <pre><code class="language-python">if label == "Left":
    distance = math.hypot(x_index - x_thumb, y_index - y_thumb)
    brush_thickness = int(np.interp(distance, [20, 200], [1, 50]))
    center_x = (x_index + x_thumb) // 2
    center_y = (y_index + y_thumb) // 2
    cv2.circle(frame, (center_x, center_y), brush_thickness, (255, 0, 0), -1)
</code></pre>
  <p>Відстань між великим та вказівним пальцем лівої руки = товщина пензля. Малюється синє коло як індикатор.</p>
<hr>
<b>Скидання попередніх координат, якщо рука не в кадрі</b>
<pre><code class="language-python">
else:
    prev_x, prev_y = None, None
</code></pre>
<p>
<strong>Пояснення:</strong><br>
1. <code>else</code> спрацьовує, коли система не виявила рук або правої руки в кадрі.<br>
2. <code>prev_x</code> та <code>prev_y</code> обнуляються, щоб наступне малювання починалося з нової точки і не з'єднувалося з попередніми лініями.
</p>

  <b>Комбінування полотна з кадром</b>
  <pre><code class="language-python">gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
_, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
frame = cv2.bitwise_and(frame, inv)
frame = cv2.bitwise_or(frame, canvas)
</code></pre>
  <p>Схрещуємо картинку з полотном: те, що малюється — накладається на відео.</p>
  <hr>

  <b>Відображення результату</b>
  <pre><code class="language-python">cv2.imshow("AirPaint", cv2.resize(frame, (1000, 700)))

if cv2.waitKey(1) & 0xFF == 27:
    break
</code></pre>
  <p>Виводимо вікно з малюванням. Вихід з програми — клавіша Esc.</p>
  <hr>

  <b>Завершення роботи</b>
  <pre><code class="language-python">cap.release()
cv2.destroyAllWindows()
</code></pre>
  <p>Звільняємо камеру та закриваємо всі вікна.</p>
</section>
  <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">
</section>

