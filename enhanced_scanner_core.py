"""
Enhanced Pump.fun Scanner with Advanced AI Analysis
Based on research showing 98.6% failure rate and advanced rugpull detection
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import os
from colorama import init, Fore, Back, Style
import pandas as pd
from enum import Enum
import logging
from solders.pubkey import Pubkey
from solders.rpc.responses import GetTokenAccountsResponse
import websockets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize colorama
init(autoreset=True)

# Constants based on research
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
RAYDIUM_MIGRATION = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
BONDING_CURVE_COMPLETION_MCAP = 68000  # $68k-$73k range
BONDING_CURVE_SOL_REQUIRED = 85
TOKEN_TOTAL_SUPPLY = 1_000_000_000  # 1 billion
BONDING_CURVE_SUPPLY = 793_000_000  # 793 million

# Based on research: 98.6% failure rate, 93% manipulation
SUCCESS_RATE = 0.014  # 1.4% success rate
MANIPULATION_RATE = 0.93  # 93% show manipulation

class RiskLevel(Enum):
    SAFE = "safe"
    MEDIUM = "medium"
    HIGH = "high"
    RUGPULL = "rugpull"

@dataclass
class TokenAnalysis:
    """Enhanced token analysis with research-based metrics"""
    # Basic info
    mint: str
    symbol: str
    name: str
    price: float
    market_cap: float
    liquidity: float
    age_minutes: int
    creator: str
    creation_tx: str
    
    # Bonding curve specific
    bonding_curve_progress: float  # 0-100%
    bonding_curve_address: str
    sol_in_curve: float
    tokens_in_curve: float
    migration_status: str  # "active", "migrating", "completed"
    
    # Holder analysis (enhanced)
    total_holders: int
    top_10_percentage: float
    top_holder_percentage: float
    unique_holders: int
    bundled_wallets: List[str]
    dev_holding: float
    suspected_insiders: List[str]
    holder_concentration_score: float  # Gini coefficient
    
    # Liquidity analysis
    liquidity_locked: bool
    liquidity_lock_duration: int  # days
    lp_burn_percentage: float
    pool_creation_time: datetime
    
    # Technical analysis
    support_levels: List[float]
    resistance_levels: List[float]
    current_rsi: float
    volume_24h: float
    volume_spike: bool
    price_change_1h: float
    price_change_24h: float
    buy_sell_ratio: float
    
    # Enhanced risk analysis
    risk_level: RiskLevel
    rugpull_probability: float
    bundle_detected: bool
    honeypot_risk: bool
    manipulation_score: float  # 0-1
    similar_creator_rugs: int
    
    # AI scoring
    pump_score: float
    success_probability: float
    recommended_action: str
    confidence_level: float  # Model confidence
    reasons: List[str]
    
    # Pattern detection
    wash_trading_detected: bool
    fake_volume_percentage: float
    bot_activity_score: float
    organic_growth_score: float
    
    # Similar token analysis
    similar_tokens_performance: Dict[str, float] = field(default_factory=dict)
    creator_history: Dict[str, any] = field(default_factory=dict)

class AdvancedAIAnalyzer:
    """Machine Learning powered analyzer with enhanced features"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.prediction_history = deque(maxlen=1000)
        self.model_version = "2.0"
        self.load_or_train_model()
        
        # Enhanced pattern database
        self.pattern_database = {
            'rugpull_patterns': [],
            'successful_patterns': [],
            'manipulation_patterns': [],
            'organic_growth_patterns': []
        }
        
        # Creator tracking
        self.creator_database = defaultdict(lambda: {
            'tokens_created': 0,
            'rugs_count': 0,
            'success_count': 0,
            'avg_lifespan': 0,
            'total_volume': 0
        })
        
    def load_or_train_model(self):
        """Load existing model or train new one"""
        model_path = f"pump_analyzer_model_v{self.model_version}.pkl"
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(f"pump_analyzer_scaler_v{self.model_version}.pkl")
            logging.info(f"Loaded existing model v{self.model_version}")
        else:
            self.train_initial_model()
    
    def train_initial_model(self):
        """Train initial model with synthetic data based on research"""
        # Generate synthetic training data based on research
        n_samples = 10000
        
        # Features based on research findings
        features = []
        labels = []
        
        for _ in range(n_samples):
            # Simulate token characteristics
            is_rug = np.random.random() < MANIPULATION_RATE
            
            if is_rug:
                # Rugpull characteristics
                liquidity = np.random.uniform(100, 5000)
                holder_concentration = np.random.uniform(70, 95)
                dev_holding = np.random.uniform(15, 40)
                unique_holders = np.random.randint(10, 100)
                age_minutes = np.random.randint(5, 120)
                volume_liquidity_ratio = np.random.uniform(0.1, 2)
                organic_score = np.random.uniform(0, 0.3)
            else:
                # Legitimate token characteristics
                liquidity = np.random.uniform(5000, 100000)
                holder_concentration = np.random.uniform(30, 70)
                dev_holding = np.random.uniform(0, 15)
                unique_holders = np.random.randint(100, 1000)
                age_minutes = np.random.randint(60, 1440)
                volume_liquidity_ratio = np.random.uniform(2, 10)
                organic_score = np.random.uniform(0.5, 1)
            
            features.append([
                liquidity,
                holder_concentration,
                dev_holding,
                unique_holders,
                age_minutes,
                volume_liquidity_ratio,
                organic_score
            ])
            labels.append(0 if is_rug else 1)
        
        # Train model
        X = np.array(features)
        y = np.array(labels)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Save model
        joblib.dump(self.model, f"pump_analyzer_model_v{self.model_version}.pkl")
        joblib.dump(self.scaler, f"pump_analyzer_scaler_v{self.model_version}.pkl")
        
        logging.info("Trained and saved initial model")
    
    def extract_features(self, analysis: TokenAnalysis) -> np.ndarray:
        """Extract features for ML model"""
        features = [
            analysis.liquidity,
            analysis.top_10_percentage,
            analysis.dev_holding,
            analysis.unique_holders,
            analysis.age_minutes,
            analysis.volume_24h / max(analysis.liquidity, 1),
            analysis.organic_growth_score,
            analysis.bonding_curve_progress,
            analysis.holder_concentration_score,
            1 if analysis.liquidity_locked else 0,
            analysis.buy_sell_ratio,
            analysis.bot_activity_score,
            analysis.similar_creator_rugs,
            len(analysis.bundled_wallets),
            analysis.fake_volume_percentage
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_success(self, analysis: TokenAnalysis) -> Tuple[float, float]:
        """Predict success probability with confidence"""
        features = self.extract_features(analysis)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        success_prob = probabilities[1]  # Probability of success
        confidence = max(probabilities)  # Model confidence
        
        # Adjust based on research (98.6% fail rate)
        adjusted_success = success_prob * SUCCESS_RATE / 0.5
        
        return adjusted_success, confidence
    
    def analyze_creator_history(self, creator: str) -> Dict:
        """Analyze creator's token history"""
        history = self.creator_database[creator]
        
        if history['tokens_created'] == 0:
            return {'risk_score': 0.5, 'new_creator': True}
        
        rug_rate = history['rugs_count'] / history['tokens_created']
        avg_lifespan = history['avg_lifespan']
        
        risk_score = (rug_rate * 0.7 + 
                     (1 - min(avg_lifespan / 1440, 1)) * 0.3)
        
        return {
            'risk_score': risk_score,
            'rug_rate': rug_rate,
            'avg_lifespan': avg_lifespan,
            'total_tokens': history['tokens_created'],
            'new_creator': False
        }
    
    def detect_manipulation_patterns(self, analysis: TokenAnalysis) -> float:
        """Detect market manipulation patterns"""
        manipulation_score = 0
        
        # Check for wash trading
        if analysis.wash_trading_detected:
            manipulation_score += 0.3
        
        # Check for fake volume
        if analysis.fake_volume_percentage > 30:
            manipulation_score += 0.2
        
        # Check for bot activity
        if analysis.bot_activity_score > 0.7:
            manipulation_score += 0.2
        
        # Check for bundle activity
        if len(analysis.bundled_wallets) > 5:
            manipulation_score += 0.3
        
        return min(manipulation_score, 1.0)

class EnhancedHolderAnalyzer:
    """Advanced holder analysis with bundle detection"""
    
    def __init__(self, rpc_client):
        self.rpc_client = rpc_client
        self.known_bundles: Set[str] = set()
        self.bundle_patterns = self._load_bundle_patterns()
        
    def _load_bundle_patterns(self) -> Dict:
        """Load known bundle patterns"""
        return {
            'same_creation_time': 300,  # 5 minutes
            'similar_balances': 0.1,    # 10% similarity
            'transaction_patterns': {
                'min_linked_txs': 3,
                'time_window': 3600     # 1 hour
            }
        }
    
    async def analyze_holders(self, mint: str) -> Dict:
        """Comprehensive holder analysis"""
        holders = await self._get_token_holders(mint)
        
        if not holders:
            return self._empty_analysis()
        
        # Sort by balance
        holders_sorted = sorted(holders, key=lambda x: x['balance'], reverse=True)
        
        # Calculate metrics
        total_supply = sum(h['balance'] for h in holders)
        top_10_balance = sum(h['balance'] for h in holders_sorted[:10])
        top_holder_balance = holders_sorted[0]['balance'] if holders_sorted else 0
        
        # Detect bundles
        bundled_wallets = await self._detect_bundles(holders_sorted)
        
        # Calculate Gini coefficient
        gini = self._calculate_gini_coefficient([h['balance'] for h in holders])
        
        # Find developer wallet
        dev_wallet, dev_holding = await self._find_dev_wallet(mint, holders_sorted)
        
        return {
            'total_holders': len(holders),
            'unique_holders': len(holders) - len(bundled_wallets),
            'top_10_percentage': (top_10_balance / total_supply) * 100,
            'top_holder_percentage': (top_holder_balance / total_supply) * 100,
            'bundled_wallets': list(bundled_wallets),
            'dev_wallet': dev_wallet,
            'dev_holding': dev_holding,
            'holder_concentration_score': gini,
            'distribution_analysis': self._analyze_distribution(holders_sorted)
        }
    
    async def _get_token_holders(self, mint: str) -> List[Dict]:
        """Get all token holders using getTokenLargestAccounts"""
        try:
            mint_pubkey = Pubkey.from_string(mint)
            
            # Get largest accounts (top 20)
            response = await self.rpc_client.get_token_largest_accounts(
                mint_pubkey,
                commitment="finalized"
            )
            
            if not response or not response.value:
                return []
            
            holders = []
            for account in response.value:
                holders.append({
                    'address': str(account.address),
                    'balance': account.amount.ui_amount or 0,
                    'decimals': account.amount.decimals
                })
            
            return holders
            
        except Exception as e:
            logging.error(f"Error getting token holders: {e}")
            return []
    
    async def _detect_bundles(self, holders: List[Dict]) -> Set[str]:
        """Detect bundled wallets using advanced heuristics"""
        bundles = set()
        
        # Check for wallets created at similar times
        for i, holder in enumerate(holders[:20]):  # Check top 20
            for j, other in enumerate(holders[i+1:21]):
                # Similar balance check
                if abs(holder['balance'] - other['balance']) / holder['balance'] < 0.1:
                    # Further analysis needed
                    if await self._are_wallets_linked(holder['address'], other['address']):
                        bundles.add(holder['address'])
                        bundles.add(other['address'])
        
        return bundles
    
    async def _are_wallets_linked(self, wallet1: str, wallet2: str) -> bool:
        """Check if two wallets are linked through transactions"""
        # Simplified check - in production, analyze transaction history
        # Look for:
        # 1. Common funding source
        # 2. Similar transaction patterns
        # 3. Transfers between wallets
        return False  # Placeholder
    
    def _calculate_gini_coefficient(self, balances: List[float]) -> float:
        """Calculate Gini coefficient for wealth distribution"""
        if not balances:
            return 0
        
        sorted_balances = sorted(balances)
        n = len(balances)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * sorted_balances)) / (n * np.sum(sorted_balances)) - (n + 1) / n
    
    async def _find_dev_wallet(self, mint: str, holders: List[Dict]) -> Tuple[str, float]:
        """Identify developer wallet and holdings"""
        # Look for the wallet that created the token
        # This would require analyzing the token creation transaction
        # For now, return the largest holder as a placeholder
        if holders:
            dev_wallet = holders[0]['address']
            total_supply = sum(h['balance'] for h in holders)
            dev_holding = (holders[0]['balance'] / total_supply) * 100
            return dev_wallet, dev_holding
        
        return "", 0
    
    def _analyze_distribution(self, holders: List[Dict]) -> Dict:
        """Analyze holder distribution patterns"""
        if not holders:
            return {'pattern': 'unknown', 'health_score': 0}
        
        # Calculate distribution metrics
        total = sum(h['balance'] for h in holders)
        
        # Check for healthy distribution
        if len(holders) > 100 and holders[0]['balance'] / total < 0.1:
            return {'pattern': 'healthy', 'health_score': 0.8}
        elif len(holders) > 50 and holders[0]['balance'] / total < 0.2:
            return {'pattern': 'moderate', 'health_score': 0.5}
        else:
            return {'pattern': 'concentrated', 'health_score': 0.2}
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'total_holders': 0,
            'unique_holders': 0,
            'top_10_percentage': 0,
            'top_holder_percentage': 0,
            'bundled_wallets': [],
            'dev_wallet': '',
            'dev_holding': 0,
            'holder_concentration_score': 0,
            'distribution_analysis': {'pattern': 'unknown', 'health_score': 0}
        }

class BondingCurveAnalyzer:
    """Analyze bonding curve progress and migration status"""
    
    def __init__(self, rpc_client):
        self.rpc_client = rpc_client
        
    async def get_bonding_curve_status(self, mint: str, bonding_curve_address: str) -> Dict:
        """Get detailed bonding curve status"""
        try:
            # Get bonding curve account data
            curve_data = await self._get_curve_data(bonding_curve_address)
            
            if not curve_data:
                return self._empty_status()
            
            # Calculate progress
            sol_raised = curve_data.get('sol_balance', 0)
            progress = (sol_raised / BONDING_CURVE_SOL_REQUIRED) * 100
            
            # Check migration status
            migration_status = await self._check_migration_status(mint, progress)
            
            return {
                'progress': min(progress, 100),
                'sol_raised': sol_raised,
                'sol_required': BONDING_CURVE_SOL_REQUIRED,
                'tokens_sold': curve_data.get('tokens_sold', 0),
                'migration_status': migration_status,
                'estimated_completion': self._estimate_completion(progress, curve_data)
            }
            
        except Exception as e:
            logging.error(f"Error getting bonding curve status: {e}")
            return self._empty_status()
    
    async def _get_curve_data(self, bonding_curve_address: str) -> Dict:
        """Get bonding curve account data"""
        # This would decode the actual bonding curve account
        # For now, return placeholder
        return {
            'sol_balance': np.random.uniform(0, 85),
            'tokens_sold': np.random.uniform(0, BONDING_CURVE_SUPPLY)
        }
    
    async def _check_migration_status(self, mint: str, progress: float) -> str:
        """Check if token is migrating to Raydium"""
        if progress < 100:
            return "active"
        elif progress == 100:
            # Check if migration transaction exists
            # This would look for withdraw instruction on migration account
            return "migrating"
        else:
            return "completed"
    
    def _estimate_completion(self, progress: float, curve_data: Dict) -> Optional[datetime]:
        """Estimate when bonding curve will complete"""
        if progress >= 100:
            return None
        
        # Simple linear estimation (would be more complex in reality)
        # Based on current progress rate
        hours_to_complete = (100 - progress) * 2  # Placeholder
        return datetime.now() + timedelta(hours=hours_to_complete)
    
    def _empty_status(self) -> Dict:
        """Return empty status structure"""
        return {
            'progress': 0,
            'sol_raised': 0,
            'sol_required': BONDING_CURVE_SOL_REQUIRED,
            'tokens_sold': 0,
            'migration_status': 'unknown',
            'estimated_completion': None
        }