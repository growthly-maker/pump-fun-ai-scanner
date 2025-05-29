"""
Configuration management for the scanner
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class Config:
    """Scanner configuration"""
    
    # RPC Settings
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    ws_url: str = "wss://api.mainnet-beta.solana.com"
    
    # Scanner Settings
    scan_interval: int = 60  # seconds
    max_concurrent_scans: int = 10
    
    # Risk Thresholds
    min_liquidity: float = 5000  # USD
    min_holders: int = 100
    max_dev_holding: float = 15  # percentage
    max_top10_holding: float = 70  # percentage
    min_unique_holders: int = 50
    
    # Trading Parameters
    pump_score_threshold: float = 70  # minimum score to consider
    confidence_threshold: float = 0.7  # AI confidence threshold
    
    # Bonding Curve Settings
    bonding_curve_alert_threshold: float = 80  # % completion to alert
    migration_monitoring: bool = True
    
    # Alert Settings
    enable_alerts: bool = True
    alert_methods: List[str] = None  # ["console", "discord", "telegram"]
    discord_webhook_url: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Trading Settings (if integrated with trading bot)
    enable_auto_trading: bool = False
    max_position_size: float = 1.0  # SOL
    default_buy_amount: float = 0.1  # SOL
    slippage_tolerance: float = 10  # percentage
    stop_loss_percentage: float = 20
    take_profit_percentage: float = 100
    
    # Advanced Settings
    enable_ml_predictions: bool = True
    ml_model_version: str = "2.0"
    pattern_detection_enabled: bool = True
    creator_tracking_enabled: bool = True
    
    # Data Storage
    save_analysis_history: bool = True
    max_history_size: int = 10000
    export_format: str = "csv"  # csv, json
    
    # Performance
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "scanner.log"
    
    def __post_init__(self):
        if self.alert_methods is None:
            self.alert_methods = ["console"]
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        if self.min_liquidity < 0:
            errors.append("min_liquidity must be positive")
        
        if self.min_holders < 1:
            errors.append("min_holders must be at least 1")
        
        if not 0 <= self.max_dev_holding <= 100:
            errors.append("max_dev_holding must be between 0 and 100")
        
        if not 0 <= self.slippage_tolerance <= 100:
            errors.append("slippage_tolerance must be between 0 and 100")
        
        if self.enable_auto_trading and self.max_position_size <= 0:
            errors.append("max_position_size must be positive when auto trading is enabled")
        
        return errors
    
    def save(self, path: str = "config.json"):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str = "config.json") -> 'Config':
        """Load configuration from file"""
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    def update_from_env(self):
        """Update config from environment variables"""
        # RPC URLs
        if rpc_url := os.getenv('SOLANA_RPC_URL'):
            self.rpc_url = rpc_url
        
        if ws_url := os.getenv('SOLANA_WS_URL'):
            self.ws_url = ws_url
        
        # Trading amounts
        if buy_amount := os.getenv('DEFAULT_BUY_AMOUNT'):
            self.default_buy_amount = float(buy_amount)
        
        if slippage := os.getenv('MAX_SLIPPAGE'):
            self.slippage_tolerance = float(slippage)
        
        # Risk settings
        if min_liq := os.getenv('MIN_LIQUIDITY_USD'):
            self.min_liquidity = float(min_liq)
        
        if min_holders := os.getenv('MIN_HOLDERS'):
            self.min_holders = int(min_holders)
        
        if max_dev := os.getenv('MAX_DEV_HOLDING'):
            self.max_dev_holding = float(max_dev)
        
        # Alerts
        if discord_webhook := os.getenv('DISCORD_WEBHOOK_URL'):
            self.discord_webhook_url = discord_webhook
            if 'discord' not in self.alert_methods:
                self.alert_methods.append('discord')
        
        if telegram_token := os.getenv('TELEGRAM_BOT_TOKEN'):
            self.telegram_bot_token = telegram_token
            if telegram_chat := os.getenv('TELEGRAM_CHAT_ID'):
                self.telegram_chat_id = telegram_chat
                if 'telegram' not in self.alert_methods:
                    self.alert_methods.append('telegram')
        
        # Features
        if auto_trade := os.getenv('ENABLE_AUTO_TRADING'):
            self.enable_auto_trading = auto_trade.lower() == 'true'

def load_config(path: Optional[str] = None) -> Config:
    """Load and validate configuration"""
    config_path = path or os.getenv('CONFIG_PATH', 'config.json')
    
    # Load from file
    config = Config.load(config_path)
    
    # Override with environment variables
    config.update_from_env()
    
    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return config

# Presets for different strategies
CONSERVATIVE_CONFIG = Config(
    min_liquidity=10000,
    min_holders=200,
    max_dev_holding=10,
    pump_score_threshold=80,
    confidence_threshold=0.8,
    enable_auto_trading=False
)

MODERATE_CONFIG = Config(
    min_liquidity=5000,
    min_holders=100,
    max_dev_holding=15,
    pump_score_threshold=70,
    confidence_threshold=0.7,
    enable_auto_trading=False
)

AGGRESSIVE_CONFIG = Config(
    min_liquidity=2000,
    min_holders=50,
    max_dev_holding=20,
    pump_score_threshold=60,
    confidence_threshold=0.6,
    enable_auto_trading=False
)

def get_preset_config(preset: str) -> Config:
    """Get a preset configuration"""
    presets = {
        'conservative': CONSERVATIVE_CONFIG,
        'moderate': MODERATE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset]
    config.update_from_env()  # Still allow env overrides
    
    return config