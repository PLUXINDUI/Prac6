"""
Модуль для обработки видео и записи сегментов нарушений
"""
import os
import cv2
from datetime import datetime
from pathlib import Path

class VideoProcessor:
    """Обработчик видео и запись сегментов"""
    
    def __init__(self, buffer_seconds=10, frame_skip=2, 
                 sleep_persistence_seconds=10):
        """
        Args:
            buffer_seconds: буфер записи после исчезновения нарушения
            frame_skip: обрабатывать каждый N-й кадр
            sleep_persistence_seconds: буфер подтверждения сна
        """
        self.buffer_seconds = buffer_seconds
        self.frame_skip = frame_skip
        self.sleep_persistence_seconds = sleep_persistence_seconds
        self.writer = None
        self.recording = False
    
    def setup_output_dirs(self):
        """Создает необходимые директории для выходных файлов"""
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        segments_dir = os.path.join("monitor_output", "segments", today_date)
        faces_dir = os.path.join("monitor_output", "faces", today_date)
        
        os.makedirs(segments_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)
        
        return segments_dir, faces_dir
    
    def start_recording(self, output_path, frame_size, fps):
        """
        Начинает запись видео
        
        Args:
            output_path: путь для сохранения видеофайла
            frame_size: размер кадра (width, height)
            fps: частота кадров
        """
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            frame_size
        )
        self.recording = True
    
    def write_frame(self, frame):
        """Записывает кадр в видеофайл"""
        if self.writer and self.recording:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Останавливает запись видео"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.recording = False
    
    def generate_segment_filename(self):
        """Генерирует имя файла сегмента на основе времени"""
        timestamp = datetime.now().strftime("%H-%M-%S")
        return f"seg_{timestamp}.mp4"
    
    def generate_report(self, segments_data, face_recognizer=None):
        """
        Генерирует отчет о нарушениях
        
        Args:
            segments_data: список словарей с данными сегментов
            face_recognizer: объект FaceRecognizer для анализа лиц
        """
        if not segments_data:
            return None
        
        today_date = datetime.now().strftime("%Y-%m-%d")
        report_dir = os.path.join("monitor_output")
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f"report_{today_date}.txt")
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write(f"ОТЧЕТ О НАРУШЕНИЯХ | ДАТА: {today_date}\n")
            f.write("="*70 + "\n\n")
        
        # Обработка каждого сегмента
        for i, item in enumerate(segments_data, 1):
            student_info = "Не опознан"
            face_path = "Нет лица"
            
            # Анализ лиц (если доступен)
            if face_recognizer and face_recognizer.is_database_available():
                try:
                    name, score, face_path = face_recognizer.analyze_video_segment(
                        item['path']
                    )
                    student_info = f"{name} ({score:.0%})" if score >= 0.5 else "Не опознан"
                except Exception:
                    student_info = "Ошибка анализа"
            
            # Запись в отчет
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(f"№{i}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Время:       {item['time']}\n")
                f.write(f"Нарушение:   {item['violation']}\n")
                f.write(f"Студент:     {student_info}\n")
                f.write(f"Видеофайл:   {item['path']}\n")
                f.write(f"Фото лица:   {face_path}\n")
                f.write("\n")
        
        return report_file
