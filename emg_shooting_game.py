import sys
import time
import random
import math

from collections import deque

import pygame
import serial
import numpy as np

# ===================== Configuration =====================
SERIAL_PORT = "/dev/ttyUSB0"  # Update to your actual port (e.g., "COM3" on Windows)
BAUDRATE = 115200

# Match ESP32 sample rate
SAMPLE_RATE = 500
WINDOW_SECONDS = 0.20
WINDOW_SIZE = max(10, int(SAMPLE_RATE * WINDOW_SECONDS))

# EMG Thresholds (adjust based on your calibration data)
ON_THRESHOLD = 550.0
OFF_THRESHOLD = 500.0
BASELINE_ALPHA = 0.01

# Game window settings - 1920x1080 resolution
WIDTH, HEIGHT = 1920, 900
FPS = 60

# Color definitions
BG_COLOR = (25, 30, 40)
PANEL_BG = (15, 20, 28)
WHITE = (255, 255, 255)
RED = (235, 80, 80)
GREEN = (70, 210, 100)
BLUE = (70, 150, 250)
YELLOW = (255, 215, 0)
GRAY = (120, 130, 140)
DARK_GRAY = (50, 60, 70)
WAVE_COLOR = (80, 220, 250)


def open_serial():
    try:
        port = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.001)
        port.reset_input_buffer()
        print(f"Successfully connected to ESP32 serial port: {SERIAL_PORT}")
        return port
    except serial.SerialException as exc:
        print(f"Could not open port {SERIAL_PORT}: {exc}")
        print(
            "Tip: Check your connection and ensure the Arduino Serial Monitor is closed."
        )
        sys.exit(1)

class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self, screen, font, mouse_pos):
        hover = self.rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, BLUE if hover else DARK_GRAY, self.rect, border_radius=12)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=12)
        label = font.render(self.text, True, WHITE)
        screen.blit(label, label.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )


def spawn_confetti(n=180):
    colors = [RED, GREEN, BLUE, YELLOW, WAVE_COLOR, WHITE]
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

class EMGController:
    """ESP32 serial EMG signal processing and rising-edge detection"""

    def __init__(self, port):
        self.port = port
        self.samples = deque(maxlen=WINDOW_SIZE)
        self.baseline = None
        self.rms = 0.0
        self.contracting = False
        self.valid_lines = 0
        self.bad_lines = 0

    def update(self):
        latest_val = None
        for _ in range(100):
            try:
                if self.port.in_waiting <= 0:
                    break
                line = self.port.readline().decode("utf-8", errors="ignore").strip()
                value = int(line)
                if not 0 <= value <= 4095:
                    self.bad_lines += 1
                    continue
                latest_val = value
            except (ValueError, serial.SerialException):
                self.bad_lines += 1
                continue

            if self.baseline is None:
                self.baseline = float(value)
            self.baseline += BASELINE_ALPHA * (value - self.baseline)
            self.samples.append(value - self.baseline)
            self.valid_lines += 1

        if len(self.samples) < WINDOW_SIZE:
            return False, latest_val

        x = np.asarray(self.samples, dtype=float)
        x = x - np.mean(x)
        self.rms = float(np.sqrt(np.mean(x * x)))

        was_contracting = self.contracting
        if not self.contracting and self.rms >= ON_THRESHOLD:
            self.contracting = True
        elif self.contracting and self.rms <= OFF_THRESHOLD:
            self.contracting = False

        # Detect EMG rising edge
        rising_edge = self.contracting and not was_contracting
        return rising_edge, latest_val


def main():
    pygame.init()
    pygame.display.set_caption("ESP32 EMG Shooting Game")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 26)
    title_font = pygame.font.SysFont(None, 64)
    big_font = pygame.font.SysFont(None, 140)

    port = open_serial()
    emg = EMGController(port)

    # ---------- Game state ----------
    state = "setup"          # "setup" | "playing" | "won"
    goal = 10                # winning score (set by mouse)
    score = 0
    misses = 0
    hits = 0
    start_time = 0.0
    elapsed = 0.0
    confetti = []

    game_area_height = 650

    crosshair_x = WIDTH // 2
    crosshair_y = game_area_height // 2
    crosshair_speed = 6.0
    crosshair_min_x = 200
    crosshair_max_x = WIDTH - 200

    target_x = WIDTH // 3
    target_y = game_area_height // 2
    target_speed = 2.0
    target_min_x = 300
    target_max_x = WIDTH - 300
    target_radius = 40
    crosshair_radius = 50

    COOLDOWN_TIME = 0.4
    last_shot_time = 0.0

    feedback_text = ""
    feedback_timer = 0.0
    feedback_color = WHITE

    waveform_buffer = deque(maxlen=800)

    # ---------- Buttons ----------
    cx = WIDTH // 2
    btn_minus = Button((cx - 260, 250, 80, 80), "-")
    btn_plus = Button((cx + 180, 250, 80, 80), "+")
    presets = [
        (n, Button((cx - 230 + i * 120, 380, 100, 60), str(n)))
        for i, n in enumerate([5, 10, 20, 30])
    ]
    btn_start = Button((cx - 150, 480, 300, 80), "START")
    btn_menu = Button((WIDTH - 220, 25, 180, 50), "Menu")
    btn_again = Button((cx - 330, 560, 300, 80), "Play Again")
    btn_change = Button((cx + 30, 560, 300, 80), "Change Goal")

    def reset_round():
        nonlocal score, misses, hits, start_time, feedback_timer, last_shot_time
        score = misses = hits = 0
        feedback_timer = 0.0
        start_time = time.monotonic()
        last_shot_time = start_time  # avoid an accidental shot right at start

    running = True
    last_input_time = time.monotonic()

    try:
        while running:
            dt = clock.tick(FPS) / 1000.0
            flap_edge, raw_val = emg.update()

            if raw_val is not None:
                waveform_buffer.append(raw_val)

            if emg.valid_lines > 0:
                last_input_time = time.monotonic()
            no_signal_seconds = time.monotonic() - last_input_time

            current_time = time.monotonic()
            fire_triggered = False
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
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        if current_time - last_shot_time >= COOLDOWN_TIME:
                            last_shot_time = current_time
                            fire_triggered = True
                    elif btn_menu.clicked(event):
                        state = "setup"

                elif state == "won":
                    if btn_again.clicked(event):
                        reset_round()
                        state = "playing"
                    elif btn_change.clicked(event):
                        state = "setup"

            # EMG rising-edge trigger (only meaningful while playing)
            if flap_edge and (current_time - last_shot_time >= COOLDOWN_TIME):
                last_shot_time = current_time
                fire_triggered = True

            # -------------------- Game Logic --------------------
            crosshair_x += crosshair_speed
            if crosshair_x >= crosshair_max_x or crosshair_x <= crosshair_min_x:
                crosshair_speed = -crosshair_speed

            target_x += target_speed
            if target_x >= target_max_x or target_x <= target_min_x:
                target_speed = -target_speed

            if fire_triggered and state == "playing":
                if abs(crosshair_x - target_x) <= crosshair_radius:
                    score += 1
                    hits += 1
                    feedback_text = "HIT!"
                    feedback_color = GREEN
                else:
                    misses += 1
                    score = max(0, score - 1)
                    feedback_text = "MISS! (-1 Pt)"
                    feedback_color = RED
                feedback_timer = 0.8

                if score >= goal:
                    state = "won"
                    elapsed = time.monotonic() - start_time
                    confetti = spawn_confetti()

            if feedback_timer > 0:
                feedback_timer -= dt

            # -------------------- Rendering --------------------
            screen.fill(BG_COLOR)
            pygame.draw.rect(screen, BG_COLOR, (0, 0, WIDTH, game_area_height))

            for gx in range(200, WIDTH, 200):
                pygame.draw.line(screen, DARK_GRAY, (gx, 0), (gx, game_area_height), 1)
            pygame.draw.line(
                screen, GRAY, (0, game_area_height // 2), (WIDTH, game_area_height // 2), 1
            )

            # Target
            pygame.draw.circle(screen, RED, (int(target_x), int(target_y)), target_radius)
            pygame.draw.circle(
                screen, WHITE, (int(target_x), int(target_y)), target_radius - 10, 2
            )
            pygame.draw.circle(screen, RED, (int(target_x), int(target_y)), 8)

            # Crosshair
            pygame.draw.circle(
                screen, BLUE, (int(crosshair_x), int(crosshair_y)), crosshair_radius, 3
            )
            pygame.draw.line(
                screen, BLUE,
                (int(crosshair_x) - crosshair_radius - 10, int(crosshair_y)),
                (int(crosshair_x) + crosshair_radius + 10, int(crosshair_y)), 3,
            )
            pygame.draw.line(
                screen, BLUE,
                (int(crosshair_x), int(crosshair_y) - crosshair_radius - 10),
                (int(crosshair_x), int(crosshair_y) + crosshair_radius + 10), 3,
            )

            if feedback_timer > 0:
                fb_surface = title_font.render(feedback_text, True, feedback_color)
                screen.blit(fb_surface, (WIDTH // 2 - fb_surface.get_width() // 2, 100))

            # HUD
            screen.blit(font.render(f"Score: {score} / {goal}", True, WHITE), (40, 30))
            screen.blit(font.render(f"Misses: {misses}", True, RED), (40, 75))

            emg_status_str = "CONTRACTION" if emg.contracting else "REST"
            emg_color = GREEN if emg.contracting else GRAY
            screen.blit(
                font.render(f"EMG Status: {emg_status_str}", True, emg_color), (40, 130)
            )
            screen.blit(
                small_font.render(
                    f"RMS: {emg.rms:.1f} | ON >= {ON_THRESHOLD:.0f} | OFF <= {OFF_THRESHOLD:.0f}",
                    True, GRAY,
                ),
                (40, 175),
            )

            # Progress bar toward the goal
            bar_rect = pygame.Rect(40, 205, 400, 18)
            pygame.draw.rect(screen, DARK_GRAY, bar_rect, border_radius=9)
            fill_w = int(bar_rect.width * min(1.0, score / max(1, goal)))
            if fill_w > 0:
                pygame.draw.rect(
                    screen, GREEN, (bar_rect.x, bar_rect.y, fill_w, bar_rect.height),
                    border_radius=9,
                )
            pygame.draw.rect(screen, GRAY, bar_rect, 1, border_radius=9)

            # Timer + Menu button (while playing)
            if state == "playing":
                t = current_time - start_time
                screen.blit(font.render(f"Time: {t:.1f}s", True, YELLOW), (40, 235))
                btn_menu.draw(screen, font, mouse_pos)

            # Waveform panel
            wave_panel_y = game_area_height + 20
            wave_panel_h = HEIGHT - wave_panel_y - 50
            pygame.draw.rect(screen, PANEL_BG, (30, wave_panel_y, WIDTH - 60, wave_panel_h))
            pygame.draw.rect(screen, GRAY, (30, wave_panel_y, WIDTH - 60, wave_panel_h), 1)
            screen.blit(
                small_font.render(
                    "Real-time EMG Signal Waveform (ESP32 Raw / Baseline Corrected)",
                    True, YELLOW,
                ),
                (50, wave_panel_y + 15),
            )

            plot_x_start = 60
            plot_y_start = wave_panel_y + 50
            plot_w = WIDTH - 120
            plot_h = wave_panel_h - 70

            for i in range(4):
                gy = plot_y_start + (plot_h / 3) * i
                pygame.draw.line(
                    screen, DARK_GRAY, (plot_x_start, int(gy)),
                    (plot_x_start + plot_w, int(gy)), 1,
                )

            if len(waveform_buffer) > 1:
                points = []
                dx = plot_w / max(1, waveform_buffer.maxlen - 1)
                for idx, val in enumerate(waveform_buffer):
                    px = plot_x_start + idx * dx
                    norm_val = max(0.0, min(4095.0, float(val)))
                    py = plot_y_start + plot_h - (norm_val / 4095.0) * plot_h
                    points.append((px, py))
                if len(points) > 1:
                    pygame.draw.aalines(screen, WAVE_COLOR, False, points)

            screen.blit(
                small_font.render(
                    "Rule: Contract muscle (or Space) when target is inside crosshair! (-1 pt on miss) | Esc: Quit",
                    True, GRAY,
                ),
                (40, HEIGHT - 35),
            )

            if no_signal_seconds > 2.0:
                warning = font.render(
                    "WARNING: No serial data detected, check ESP32 connection!", True, RED
                )
                screen.blit(warning, (WIDTH // 2 - warning.get_width() // 2, 250))

            # -------------------- Overlays --------------------
            if state == "setup":
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((10, 14, 22, 220))
                screen.blit(overlay, (0, 0))

                t = title_font.render("Set Winning Score", True, WHITE)
                screen.blit(t, (cx - t.get_width() // 2, 150))
                g = big_font.render(str(goal), True, YELLOW)
                screen.blit(g, g.get_rect(center=(cx, 290)))

                btn_minus.draw(screen, title_font, mouse_pos)
                btn_plus.draw(screen, title_font, mouse_pos)
                for n, b in presets:
                    b.draw(screen, font, mouse_pos)
                btn_start.draw(screen, title_font, mouse_pos)

                hint = small_font.render(
                    "Click  - / +  or a preset (mouse wheel works too), then press START (or Enter)",
                    True, GRAY,
                )
                screen.blit(hint, (cx - hint.get_width() // 2, 590))

            elif state == "won":
                # Falling confetti
                for p in confetti:
                    p["x"] += p["vx"] * dt
                    p["y"] += p["vy"] * dt
                    if p["y"] > HEIGHT:
                        p["y"] = -10
                        p["x"] = random.uniform(0, WIDTH)

                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((10, 14, 22, 200))
                screen.blit(overlay, (0, 0))

                for p in confetti:
                    pygame.draw.rect(
                        screen, p["color"], (int(p["x"]), int(p["y"]), p["size"], p["size"])
                    )

                bob = int(8 * math.sin(current_time * 4))
                c1 = big_font.render("CONGRATULATIONS!", True, YELLOW)
                screen.blit(c1, c1.get_rect(center=(cx, 220 + bob)))

                c2 = title_font.render(f"You reached {goal} points!", True, GREEN)
                screen.blit(c2, c2.get_rect(center=(cx, 340)))

                shots = hits + misses
                acc = hits / shots * 100 if shots else 0.0
                c3 = font.render(
                    f"Time: {elapsed:.1f}s   |   Hits: {hits}   Misses: {misses}   Accuracy: {acc:.0f}%",
                    True, WHITE,
                )
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
