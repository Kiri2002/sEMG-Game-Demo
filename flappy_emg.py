import sys
import time
import random
from collections import deque
import math
import pygame
import serial
import numpy as np

# ===================== Configuration =====================
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

# Match the ESP32 sketch: analogRead(GPIO34), one integer per line.
SAMPLE_RATE = 500
WINDOW_SECONDS = 0.20
WINDOW_SIZE = max(10, int(SAMPLE_RATE * WINDOW_SECONDS))

# Initial thresholds only. Prefer values learned from your calibration data.
ON_THRESHOLD = 550.0
OFF_THRESHOLD = 500.0

# Baseline tracking.
BASELINE_ALPHA = 0.01

# Game settings (Updated for 1920x900)
WIDTH, HEIGHT = 1920, 900
FPS = 60

# 匀速控制参数
RISE_SPEED = -4.5  # 上升匀速速度 (负数表示向上)
FALL_SPEED = 3.5  # 自由下落固定匀速速度 (正数表示向下)
FLAP_DURATION = 0.22  # 每次触发向上匀速上升持续的时间 (秒)

PIPE_WIDTH = 90
PIPE_GAP = 320  # 1. 管道间的口开大些 (从240增大到320)
PIPE_SPEED = 3.5  # 适合宽屏的管道移动速度
PIPE_SPACING = 450  # 管道生成间距
BIRD_X = 300  # 小鸟水平位置
BIRD_RADIUS = 15

BG = (150, 215, 245)
GREEN = (63, 175, 86)
DARK_GREEN = (35, 125, 58)
YELLOW = (255, 218, 72)
WHITE = (255, 255, 255)
BLACK = (35, 45, 55)
RED = (220, 70, 70)


def open_serial():
    try:
        port = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.001)
        port.reset_input_buffer()
        print(f"Connected to ESP32 on {SERIAL_PORT}")
        return port
    except serial.SerialException as exc:
        print(f"Could not open {SERIAL_PORT}: {exc}")
        print("Check the port and close Arduino Serial Monitor/Plotter.")
        sys.exit(1)


class EMGController:
    def __init__(self, port):
        self.port = port
        self.samples = deque(maxlen=WINDOW_SIZE)
        self.rms_history = deque(maxlen=200)  # 2. 用于绘制波形图的历史RMS数据
        self.baseline = None
        self.rms = 0.0
        self.contracting = False
        self.last_flap = False
        self.valid_lines = 0
        self.bad_lines = 0

    def update(self):
        # Drain available serial data, keeping only valid integer samples.
        for _ in range(100):
            try:
                if self.port.in_waiting <= 0:
                    break
                line = self.port.readline().decode("utf-8", errors="ignore").strip()
                value = int(line)
                if not 0 <= value <= 4095:
                    self.bad_lines += 1
                    continue
            except (ValueError, serial.SerialException):
                self.bad_lines += 1
                continue

            if self.baseline is None:
                self.baseline = float(value)
            self.baseline += BASELINE_ALPHA * (value - self.baseline)
            self.samples.append(value - self.baseline)
            self.valid_lines += 1

        if len(self.samples) < WINDOW_SIZE:
            return False

        x = np.asarray(self.samples, dtype=float)
        x = x - np.mean(x)
        self.rms = float(np.sqrt(np.mean(x * x)))

        # 记录RMS用于波形图显示
        self.rms_history.append(self.rms)

        was_contracting = self.contracting
        if not self.contracting and self.rms >= ON_THRESHOLD:
            self.contracting = True
        elif self.contracting and self.rms <= OFF_THRESHOLD:
            self.contracting = False

        rising_edge = self.contracting and not was_contracting
        return rising_edge


def make_pipe(x):
    margin = 100
    gap_center = random.randint(margin + PIPE_GAP // 2, HEIGHT - margin - PIPE_GAP // 2)
    return {"x": float(x), "center": gap_center, "passed": False}


def circle_rect_collision(cx, cy, radius, rect):
    closest_x = max(rect.left, min(cx, rect.right))
    closest_y = max(rect.top, min(cy, rect.bottom))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius

class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, screen, font, mouse_pos):
        hover = self.rect.collidepoint(mouse_pos)
        pygame.draw.rect(
            screen, DARK_GREEN if hover else GREEN, self.rect, border_radius=12
        )
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=12)
        label = font.render(self.text, True, WHITE)
        screen.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


def spawn_confetti(n=180):
    colors = [RED, GREEN, YELLOW, WHITE, (0, 100, 255), (255, 140, 0)]
    return [
        {
            "x": random.uniform(0, WIDTH),
            "y": random.uniform(-HEIGHT, 0),
            "vx": random.uniform(-60, 60),
            "vy": random.uniform(150, 350),
            "size": random.randint(6, 14),
            "color": random.choice(colors),
        }
        for _ in range(n)
    ]

def main():
    pygame.init()
    pygame.display.set_caption("EMG Flappy Bird — Contract to Flap")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)
    title_font = pygame.font.SysFont(None, 68)
    big_font = pygame.font.SysFont(None, 140)

    port = open_serial()
    emg = EMGController(port)

    # ---------- Game state ----------
    state = "setup"  # "setup" | "playing" | "won"
    goal = 10        # winning score (set by mouse)

    bird_y = HEIGHT * 0.48
    flap_timer = 0.0
    pipes = [make_pipe(WIDTH + 100), make_pipe(WIDTH + 100 + PIPE_SPACING)]
    score = 0
    game_over = False
    paused = False
    play_time = 0.0   # seconds of actual play (excludes pause / game over)
    elapsed = 0.0
    confetti = []

    last_input_time = time.monotonic()
    no_signal_seconds = 0.0
    running = True

    # ---------- Buttons ----------
    cx = WIDTH // 2
    btn_minus = Button((cx - 260, 250, 80, 80), "-")
    btn_plus = Button((cx + 180, 250, 80, 80), "+")
    presets = [
        (n, Button((cx - 230 + i * 120, 380, 100, 60), str(n)))
        for i, n in enumerate([5, 10, 20, 30])
    ]
    btn_start = Button((cx - 150, 480, 300, 80), "START")
    btn_menu = Button((WIDTH - 600, 25, 200, 50), "Menu")
    btn_retry = Button((cx - 230, HEIGHT // 2 + 90, 200, 60), "Retry")
    btn_over_menu = Button((cx + 30, HEIGHT // 2 + 90, 200, 60), "Menu")
    btn_again = Button((cx - 330, 560, 300, 80), "Play Again")
    btn_change = Button((cx + 30, 560, 300, 80), "Change Goal")

    def reset_round():
        nonlocal bird_y, flap_timer, pipes, score, game_over, paused, play_time
        bird_y = HEIGHT * 0.48
        flap_timer = 0.0
        pipes = [make_pipe(WIDTH + 100), make_pipe(WIDTH + 100 + PIPE_SPACING)]
        score = 0
        game_over = False
        paused = False
        play_time = 0.0

    try:
        while running:
            dt = min(clock.tick(FPS) / (1000.0 / FPS), 2.0)  # frame units
            dt_s = dt / FPS                                    # seconds
            flap_edge = emg.update()

            if emg.valid_lines > 0:
                last_input_time = time.monotonic()
            no_signal_seconds = time.monotonic() - last_input_time

            mouse_pos = pygame.mouse.get_pos()

            # -------------------- Events --------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    continue

                if state == "setup":
                    if btn_minus.clicked(event):
                        goal = max(1, goal - 1)
                    elif btn_plus.clicked(event):
                        goal = min(99, goal + 1)
                    elif event.type == pygame.MOUSEWHEEL:
                        goal = max(1, min(99, goal + event.y))
                    else:
                        for n, b in presets:
                            if b.clicked(event):
                                goal = n
                    if btn_start.clicked(event) or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN
                    ):
                        reset_round()
                        state = "playing"

                elif state == "playing":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            if game_over:
                                reset_round()
                            else:
                                flap_timer = FLAP_DURATION
                        elif event.key == pygame.K_r and game_over:
                            reset_round()
                        elif event.key == pygame.K_p:
                            paused = not paused
                    elif game_over and btn_retry.clicked(event):
                        reset_round()
                    elif game_over and btn_over_menu.clicked(event):
                        state = "setup"
                    elif not game_over and btn_menu.clicked(event):
                        state = "setup"

                elif state == "won":
                    if btn_again.clicked(event):
                        reset_round()
                        state = "playing"
                    elif btn_change.clicked(event):
                        state = "setup"

            # -------------------- Game Logic --------------------
            if state == "playing" and not paused and not game_over:
                play_time += dt_s

                if flap_edge:
                    flap_timer = FLAP_DURATION

                if flap_timer > 0:
                    bird_v = RISE_SPEED
                    flap_timer -= dt_s
                else:
                    bird_v = FALL_SPEED

                bird_y += bird_v * dt

                for pipe in pipes:
                    pipe["x"] -= PIPE_SPEED * dt
                    if not pipe["passed"] and pipe["x"] + PIPE_WIDTH < BIRD_X:
                        pipe["passed"] = True
                        score += 1

                if pipes and pipes[-1]["x"] < WIDTH - PIPE_SPACING:
                    pipes.append(make_pipe(pipes[-1]["x"] + PIPE_SPACING))

                pipes = [p for p in pipes if p["x"] + PIPE_WIDTH > -10]

                if bird_y - BIRD_RADIUS <= 0 or bird_y + BIRD_RADIUS >= HEIGHT:
                    game_over = True

                for pipe in pipes:
                    x = int(pipe["x"])
                    gap_top = int(pipe["center"] - PIPE_GAP / 2)
                    gap_bottom = int(pipe["center"] + PIPE_GAP / 2)
                    top_rect = pygame.Rect(x, 0, PIPE_WIDTH, gap_top)
                    bottom_rect = pygame.Rect(
                        x, gap_bottom, PIPE_WIDTH, HEIGHT - gap_bottom
                    )
                    if circle_rect_collision(
                        BIRD_X, bird_y, BIRD_RADIUS, top_rect
                    ) or circle_rect_collision(
                        BIRD_X, bird_y, BIRD_RADIUS, bottom_rect
                    ):
                        game_over = True

                # Win check (a collision on the same frame still counts as a loss)
                if not game_over and score >= goal:
                    state = "won"
                    elapsed = play_time
                    confetti = spawn_confetti()

            # -------------------- Draw --------------------
            screen.fill(BG)

            for cloud_x, cloud_y in [
                (150, 100), (550, 140), (980, 80), (1400, 120), (1750, 90),
            ]:
                pygame.draw.circle(screen, WHITE, (cloud_x, cloud_y), 28)
                pygame.draw.circle(screen, WHITE, (cloud_x + 30, cloud_y + 5), 22)
                pygame.draw.circle(screen, WHITE, (cloud_x - 25, cloud_y + 10), 20)

            for pipe in pipes:
                x = int(pipe["x"])
                gap_top = int(pipe["center"] - PIPE_GAP / 2)
                gap_bottom = int(pipe["center"] + PIPE_GAP / 2)
                pygame.draw.rect(screen, GREEN, (x, 0, PIPE_WIDTH, gap_top))
                pygame.draw.rect(
                    screen, GREEN, (x, gap_bottom, PIPE_WIDTH, HEIGHT - gap_bottom)
                )
                pygame.draw.rect(screen, DARK_GREEN, (x, 0, 10, gap_top))
                pygame.draw.rect(
                    screen, DARK_GREEN, (x, gap_bottom, 10, HEIGHT - gap_bottom)
                )
                pygame.draw.rect(
                    screen, GREEN, (x - 8, gap_top - 20, PIPE_WIDTH + 16, 20)
                )
                pygame.draw.rect(
                    screen, GREEN, (x - 8, gap_bottom, PIPE_WIDTH + 16, 20)
                )

            # Bird
            pygame.draw.circle(screen, YELLOW, (BIRD_X, int(bird_y)), BIRD_RADIUS)
            pygame.draw.circle(screen, WHITE, (BIRD_X + 8, int(bird_y) - 6), 7)
            pygame.draw.circle(screen, BLACK, (BIRD_X + 11, int(bird_y) - 6), 3)
            pygame.draw.polygon(
                screen, RED,
                [
                    (BIRD_X + 15, int(bird_y) + 2),
                    (BIRD_X + 32, int(bird_y) + 8),
                    (BIRD_X + 15, int(bird_y) + 12),
                ],
            )

            # HUD
            screen.blit(font.render(f"Score: {score} / {goal}", True, BLACK), (30, 20))
            emg_state = "CONTRACTION / FLAP" if emg.contracting else "REST"
            state_color = DARK_GREEN if emg.contracting else BLACK
            screen.blit(font.render(f"EMG: {emg_state}", True, state_color), (30, 60))
            screen.blit(
                small_font.render(
                    f"RMS: {emg.rms:.1f} | ON >= {ON_THRESHOLD:.0f} | OFF <= {OFF_THRESHOLD:.0f}",
                    True, BLACK,
                ),
                (30, 100),
            )

            # Progress bar toward the goal
            bar_rect = pygame.Rect(30, 130, 360, 16)
            pygame.draw.rect(screen, (220, 230, 235), bar_rect, border_radius=8)
            fill_w = int(bar_rect.width * min(1.0, score / max(1, goal)))
            if fill_w > 0:
                pygame.draw.rect(
                    screen, GREEN, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height),
                    border_radius=8,
                )
            pygame.draw.rect(screen, BLACK, bar_rect, 1, border_radius=8)

            if state == "playing":
                screen.blit(
                    font.render(f"Time: {play_time:.1f}s", True, BLACK), (30, 160)
                )
                if not game_over:
                    btn_menu.draw(screen, font, mouse_pos)

            screen.blit(
                small_font.render(
                    f"Gap: {PIPE_GAP}px | Pipe speed: {PIPE_SPEED:.1f} | Space: test flap | P: pause | Esc: quit",
                    True, BLACK,
                ),
                (30, HEIGHT - 35),
            )

            # EMG RMS waveform panel
            panel_x, panel_y, panel_w, panel_h = WIDTH - 380, 20, 350, 130
            pygame.draw.rect(
                screen, (240, 245, 250), (panel_x, panel_y, panel_w, panel_h)
            )
            pygame.draw.rect(screen, BLACK, (panel_x, panel_y, panel_w, panel_h), 2)
            screen.blit(
                small_font.render("EMG RMS Waveform", True, BLACK),
                (panel_x + 10, panel_y + 8),
            )

            max_val = max(1000.0, max(emg.rms_history) if emg.rms_history else 1000.0)

            def val_to_y(val):
                inner_top = panel_y + 30
                inner_bottom = panel_y + panel_h - 10
                ratio = min(1.0, max(0.0, val / max_val))
                return int(inner_bottom - ratio * (inner_bottom - inner_top))

            on_y = val_to_y(ON_THRESHOLD)
            off_y = val_to_y(OFF_THRESHOLD)
            pygame.draw.line(
                screen, RED, (panel_x + 10, on_y), (panel_x + panel_w - 10, on_y), 1
            )
            pygame.draw.line(
                screen, (0, 100, 255),
                (panel_x + 10, off_y), (panel_x + panel_w - 10, off_y), 1,
            )
            if len(emg.rms_history) > 1:
                points = []
                dx_step = (panel_w - 20) / max(1, emg.rms_history.maxlen - 1)
                for i, val in enumerate(emg.rms_history):
                    points.append((panel_x + 10 + i * dx_step, val_to_y(val)))
                pygame.draw.lines(screen, DARK_GREEN, False, points, 2)

            if no_signal_seconds > 2.0:
                warning = font.render("No serial data — check ESP32/USB", True, RED)
                screen.blit(warning, (WIDTH // 2 - warning.get_width() // 2, 190))

            if state == "playing" and paused:
                label = font.render("PAUSED — press P to resume", True, BLACK)
                screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2))

            if state == "playing" and game_over:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((255, 255, 255, 170))
                screen.blit(overlay, (0, 0))
                title = title_font.render("Game Over", True, BLACK)
                info = font.render(
                    f"Score: {score} / {goal}  |  Press SPACE or R to restart", True, BLACK
                )
                screen.blit(
                    title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 50)
                )
                screen.blit(
                    info, (WIDTH // 2 - info.get_width() // 2, HEIGHT // 2 + 20)
                )
                btn_retry.draw(screen, font, mouse_pos)
                btn_over_menu.draw(screen, font, mouse_pos)

            # -------------------- Overlays --------------------
            if state == "setup":
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((255, 255, 255, 215))
                screen.blit(overlay, (0, 0))

                t = title_font.render("Set Winning Score (pipes to pass)", True, BLACK)
                screen.blit(t, (cx - t.get_width() // 2, 150))
                g = big_font.render(str(goal), True, DARK_GREEN)
                screen.blit(g, g.get_rect(center=(cx, 290)))

                btn_minus.draw(screen, title_font, mouse_pos)
                btn_plus.draw(screen, title_font, mouse_pos)
                for n, b in presets:
                    b.draw(screen, font, mouse_pos)
                btn_start.draw(screen, title_font, mouse_pos)

                hint = small_font.render(
                    "Click  - / +  or a preset (mouse wheel works too), then press START (or Enter)",
                    True, BLACK,
                )
                screen.blit(hint, (cx - hint.get_width() // 2, 590))

            elif state == "won":
                for p in confetti:
                    p["x"] += p["vx"] * dt_s
                    p["y"] += p["vy"] * dt_s
                    if p["y"] > HEIGHT:
                        p["y"] = -10
                        p["x"] = random.uniform(0, WIDTH)

                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((20, 40, 60, 175))
                screen.blit(overlay, (0, 0))

                for p in confetti:
                    pygame.draw.rect(
                        screen, p["color"], (int(p["x"]), int(p["y"]), p["size"], p["size"])
                    )

                bob = int(8 * math.sin(time.monotonic() * 4))
                c1 = big_font.render("CONGRATULATIONS!", True, YELLOW)
                screen.blit(c1, c1.get_rect(center=(cx, 220 + bob)))

                c2 = title_font.render(f"You passed {goal} pipes!", True, WHITE)
                screen.blit(c2, c2.get_rect(center=(cx, 340)))

                c3 = font.render(f"Time: {elapsed:.1f}s", True, WHITE)
                screen.blit(c3, c3.get_rect(center=(cx, 430)))

                btn_again.draw(screen, font, mouse_pos)
                btn_change.draw(screen, font, mouse_pos)

            pygame.display.flip()

    finally:
        if port.is_open:
            port.close()
        pygame.quit()


if __name__ == "__main__":
    main()

