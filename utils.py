"""
Utility functions for the scanner
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

class AlertManager:
    """Manage alerts across different channels"""
    
    def __init__(self, config):
        self.config = config
        self.alert_history = []
        
    async def send_alert(self, token_analysis):
        """Send alert through configured channels"""
        for method in self.config.alert_methods:
            if method == "console":
                self._console_alert(token_analysis)
            elif method == "discord" and self.config.discord_webhook_url:
                await self._discord_alert(token_analysis)
            elif method == "telegram" and self.config.telegram_bot_token:
                await self._telegram_alert(token_analysis)
    
    def _console_alert(self, analysis):
        """Print alert to console"""
        print(f"\n🚨 ALERT: {analysis.symbol} - Score: {analysis.pump_score:.1f}")
        print(f"   Price: ${analysis.price:.8f}")
        print(f"   Market Cap: ${analysis.market_cap:,.0f}")
        print(f"   Action: {analysis.recommended_action}")
    
    async def _discord_alert(self, analysis):
        """Send Discord webhook alert"""
        embed = {
            "title": f"🎯 {analysis.symbol} Opportunity Alert",
            "color": 0x00ff00 if analysis.recommended_action == "BUY" else 0xffff00,
            "fields": [
                {"name": "Score", "value": f"{analysis.pump_score:.1f}/100", "inline": True},
                {"name": "Price", "value": f"${analysis.price:.8f}", "inline": True},
                {"name": "Market Cap", "value": f"${analysis.market_cap:,.0f}", "inline": True},
                {"name": "Liquidity", "value": f"${analysis.liquidity:,.0f}", "inline": True},
                {"name": "Risk Level", "value": analysis.risk_level.value.upper(), "inline": True},
                {"name": "Action", "value": analysis.recommended_action, "inline": True},
                {"name": "Bonding Progress", "value": f"{analysis.bonding_curve_progress:.1f}%", "inline": True},
                {"name": "Contract", "value": f"`{analysis.mint}`", "inline": False},
                {"name": "Top Reasons", "value": "\n".join(analysis.reasons[:3]), "inline": False}
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Pump.fun AI Scanner"}
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    self.config.discord_webhook_url,
                    json={"embeds": [embed]}
                )
            except Exception as e:
                logging.error(f"Discord alert failed: {e}")
    
    async def _telegram_alert(self, analysis):
        """Send Telegram alert"""
        message = f"""🎯 *{analysis.symbol} Alert*

Score: {analysis.pump_score:.1f}/100
Price: ${analysis.price:.8f}
Market Cap: ${analysis.market_cap:,.0f}
Risk: {analysis.risk_level.value.upper()}
Action: *{analysis.recommended_action}*

Contract: `{analysis.mint}`

Reasons:
{chr(10).join(analysis.reasons[:3])}
"""
        
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(url, json={
                    "chat_id": self.config.telegram_chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                })
            except Exception as e:
                logging.error(f"Telegram alert failed: {e}")

class DataExporter:
    """Export scanner data in various formats"""
    
    @staticmethod
    def export_to_csv(tokens: Dict, filename: str = "pump_scan_results.csv"):
        """Export token data to CSV"""
        import pandas as pd
        
        data = []
        for token in tokens.values():
            data.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': token.symbol,
                'name': token.name,
                'mint': token.mint,
                'pump_score': token.pump_score,
                'risk_level': token.risk_level.value,
                'recommendation': token.recommended_action,
                'price': token.price,
                'market_cap': token.market_cap,
                'liquidity': token.liquidity,
                'volume_24h': token.volume_24h,
                'holders': token.total_holders,
                'unique_holders': token.unique_holders,
                'top_10_percentage': token.top_10_percentage,
                'dev_holding': token.dev_holding,
                'bonding_progress': token.bonding_curve_progress,
                'migration_status': token.migration_status,
                'rugpull_probability': token.rugpull_probability,
                'success_probability': token.success_probability,
                'confidence_level': token.confidence_level,
                'age_minutes': token.age_minutes,
                'bundle_detected': token.bundle_detected,
                'wash_trading': token.wash_trading_detected,
                'manipulation_score': token.manipulation_score
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def export_to_json(tokens: Dict, filename: str = "pump_scan_results.json"):
        """Export token data to JSON"""
        data = []
        for token in tokens.values():
            token_dict = {
                'timestamp': datetime.now().isoformat(),
                'basic_info': {
                    'symbol': token.symbol,
                    'name': token.name,
                    'mint': token.mint,
                    'price': token.price,
                    'market_cap': token.market_cap,
                    'age_minutes': token.age_minutes
                },
                'analysis': {
                    'pump_score': token.pump_score,
                    'risk_level': token.risk_level.value,
                    'recommendation': token.recommended_action,
                    'confidence': token.confidence_level,
                    'rugpull_probability': token.rugpull_probability,
                    'success_probability': token.success_probability
                },
                'holder_metrics': {
                    'total_holders': token.total_holders,
                    'unique_holders': token.unique_holders,
                    'top_10_percentage': token.top_10_percentage,
                    'dev_holding': token.dev_holding,
                    'bundle_detected': token.bundle_detected,
                    'bundled_wallets': token.bundled_wallets
                },
                'liquidity_metrics': {
                    'liquidity_usd': token.liquidity,
                    'volume_24h': token.volume_24h,
                    'liquidity_locked': token.liquidity_locked
                },
                'bonding_curve': {
                    'progress': token.bonding_curve_progress,
                    'sol_raised': token.sol_in_curve,
                    'migration_status': token.migration_status
                },
                'patterns': {
                    'wash_trading': token.wash_trading_detected,
                    'manipulation_score': token.manipulation_score,
                    'bot_activity': token.bot_activity_score,
                    'organic_score': token.organic_growth_score
                },
                'reasons': token.reasons
            }
            data.append(token_dict)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename

class TokenValidator:
    """Validate token data and contracts"""
    
    @staticmethod
    def is_valid_mint(mint: str) -> bool:
        """Check if mint address is valid"""
        try:
            # Basic validation
            if not mint or len(mint) != 44:
                return False
            
            # Check if it's base58
            import base58
            base58.b58decode(mint)
            return True
        except:
            return False
    
    @staticmethod
    def is_pump_fun_token(mint: str) -> bool:
        """Check if token is from pump.fun"""
        # Pump.fun tokens typically end with 'pump'
        return mint.endswith('pump')
    
    @staticmethod
    def validate_token_data(token_data: Dict) -> bool:
        """Validate token data structure"""
        required_fields = ['baseToken', 'priceUsd', 'liquidity']
        
        for field in required_fields:
            if field not in token_data:
                return False
        
        base_token = token_data.get('baseToken', {})
        if not all(k in base_token for k in ['address', 'symbol', 'name']):
            return False
        
        return True

class PerformanceTracker:
    """Track scanner performance and predictions"""
    
    def __init__(self):
        self.predictions = []
        self.outcomes = {}
        
    def record_prediction(self, mint: str, pump_score: float, recommendation: str):
        """Record a prediction"""
        self.predictions.append({
            'timestamp': datetime.now(),
            'mint': mint,
            'pump_score': pump_score,
            'recommendation': recommendation
        })
    
    def record_outcome(self, mint: str, actual_performance: float):
        """Record actual outcome"""
        self.outcomes[mint] = {
            'performance': actual_performance,
            'timestamp': datetime.now()
        }
    
    def calculate_accuracy(self) -> Dict:
        """Calculate prediction accuracy"""
        correct_predictions = 0
        total_with_outcomes = 0
        
        for pred in self.predictions:
            if pred['mint'] in self.outcomes:
                total_with_outcomes += 1
                outcome = self.outcomes[pred['mint']]
                
                # Check if prediction was correct
                if pred['recommendation'] == 'BUY' and outcome['performance'] > 0:
                    correct_predictions += 1
                elif pred['recommendation'] == 'AVOID' and outcome['performance'] < 0:
                    correct_predictions += 1
        
        accuracy = correct_predictions / total_with_outcomes if total_with_outcomes > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(self.predictions),
            'predictions_with_outcomes': total_with_outcomes,
            'correct_predictions': correct_predictions
        }

def format_number(num: float, decimals: int = 2) -> str:
    """Format number with appropriate suffix"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{decimals}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"

def calculate_time_until_migration(bonding_progress: float, creation_time: datetime) -> Optional[str]:
    """Estimate time until token migration"""
    if bonding_progress >= 100:
        return "Ready for migration"
    
    if bonding_progress == 0:
        return "Unknown"
    
    # Calculate rate of progress
    time_elapsed = (datetime.now() - creation_time).total_seconds() / 3600  # hours
    if time_elapsed == 0:
        return "Unknown"
    
    progress_rate = bonding_progress / time_elapsed  # % per hour
    
    if progress_rate == 0:
        return "Never"
    
    remaining_progress = 100 - bonding_progress
    hours_remaining = remaining_progress / progress_rate
    
    if hours_remaining < 1:
        return f"{int(hours_remaining * 60)} minutes"
    elif hours_remaining < 24:
        return f"{int(hours_remaining)} hours"
    else:
        return f"{int(hours_remaining / 24)} days"