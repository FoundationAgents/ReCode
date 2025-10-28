import os
import logging
from datetime import datetime
from pathlib import Path

class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        
        lines = msg.split('\n')
        if len(lines) <= 1:
            return msg
            
        return '\n'.join(lines)

class SimpleLogger:
    def __init__(self, run_id=None, log_level=logging.INFO):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.base_dir = Path("logs") / run_id
        self.log_dir = self.base_dir / "running_logs"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        sanitized_run_id = run_id.replace("/", "_").replace("\\", "_")
        self.logger = logging.getLogger(f"alfworld_run_{sanitized_run_id}")
        self.logger.setLevel(log_level)
        
        self.logger.handlers.clear()
        
        log_file = self.log_dir / "run.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        formatter = MultiLineFormatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info(f"Starting new run with ID: {run_id}")
        self.info(f"Logs will be saved to: {self.log_dir.absolute()}")
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def log_result(self, result):
        task_id = result.get("task_id", "unknown")
        success = result.get("both_success", result.get("is_success", False))
        exec_time = result.get("execution_time", result.get("time", 0))
        game_name = result.get("game_name", "")
        
        status = "SUCCESS" if success else "FAILED"
        self.info(f"[{status}] {task_id} - {game_name} - {exec_time:.2f}s")
        
        if "error" in result:
            self.error(f"Error in {task_id}: {result['error']}")
    
    def log_stats(self, stats):
        self.info("=" * 50)
        self.info("RUN STATISTICS")
        self.info("=" * 50)
        self.info(f"Total tests: {stats['total_tests']}")
        self.info(f"Successful: {stats['successful_tests']}")
        self.info(f"Success rate: {stats['success_rate']:.1%}")
        self.info(f"Average execution time: {stats['average_execution_time']:.2f}s")
        
        if stats.get('task_types'):
            self.info("\nSuccess rate by task type:")
            for task_type, type_stats in stats['task_types'].items():
                rate = type_stats['rate']
                total = type_stats['total']
                success = type_stats['success']
                self.info(f"  {task_type}: {success}/{total} ({rate:.1%})")
    
    def get_log_dir(self):
        return self.log_dir

    def get_base_dir(self):
        return self.base_dir 