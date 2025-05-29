"""
Main scanner class with enhanced detection capabilities
"""

from enhanced_scanner_core import *
import aiohttp
from typing import AsyncGenerator
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio

console = Console()

class EnhancedPumpScanner:
    """Main scanner with all advanced features integrated"""
    
    def __init__(self, rpc_client):
        self.rpc_client = rpc_client
        self.ai_analyzer = AdvancedAIAnalyzer()
        self.holder_analyzer = EnhancedHolderAnalyzer(rpc_client)
        self.bonding_analyzer = BondingCurveAnalyzer(rpc_client)
        
        self.scanned_tokens: Dict[str, TokenAnalysis] = {}
        self.watching_tokens: List[str] = []
        self.alert_tokens: List[str] = []
        
        # Statistics
        self.stats = {
            'tokens_scanned': 0,
            'rugs_detected': 0,
            'opportunities_found': 0,
            'tokens_migrated': 0
        }
        
    async def start_scanning(self):
        """Start all scanning tasks"""
        console.print("[bold cyan]🚀 Enhanced Pump.fun Scanner Starting...[/bold cyan]")
        console.print("[yellow]Based on research: 98.6% tokens fail, 93% show manipulation[/yellow]\n")
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.scan_dexscreener()),
            asyncio.create_task(self.monitor_new_tokens()),
            asyncio.create_task(self.monitor_migrations()),
            asyncio.create_task(self.update_watched_tokens()),
            asyncio.create_task(self.display_dashboard())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            console.print("\n[yellow]Scanner stopped[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
    
    async def scan_dexscreener(self):
        """Enhanced DexScreener scanning"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    # Get pump.fun tokens from DexScreener
                    url = "https://api.dexscreener.com/latest/dex/pairs/solana"
                    
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            pairs = data.get('pairs', [])
                            
                            # Filter pump.fun tokens
                            pump_tokens = [
                                p for p in pairs 
                                if p.get('dexId') and 'pump' in p.get('dexId', '').lower()
                            ]
                            
                            # Analyze each token
                            for token_data in pump_tokens[:30]:  # Top 30
                                try:
                                    await self.analyze_token(token_data)
                                except Exception as e:
                                    logging.error(f"Error analyzing token: {e}")
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logging.error(f"DexScreener scan error: {e}")
                await asyncio.sleep(120)
    
    async def monitor_new_tokens(self):
        """Monitor new token creations via WebSocket"""
        while True:
            try:
                async with websockets.connect('wss://api.mainnet-beta.solana.com') as ws:
                    # Subscribe to pump.fun program logs
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [PUMP_PROGRAM]},
                            {"commitment": "finalized"}
                        ]
                    }))
                    
                    async for msg in ws:
                        data = json.loads(msg)
                        if 'params' in data:
                            await self._process_new_token(data['params'])
                            
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await asyncio.sleep(30)
    
    async def monitor_migrations(self):
        """Monitor tokens migrating to Raydium"""
        while True:
            try:
                async with websockets.connect('wss://api.mainnet-beta.solana.com') as ws:
                    # Subscribe to migration contract
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [RAYDIUM_MIGRATION]},
                            {"commitment": "finalized"}
                        ]
                    }))
                    
                    async for msg in ws:
                        data = json.loads(msg)
                        if 'params' in data:
                            await self._process_migration(data['params'])
                            
            except Exception as e:
                logging.error(f"Migration monitor error: {e}")
                await asyncio.sleep(30)
    
    async def analyze_token(self, token_data: Dict) -> Optional[TokenAnalysis]:
        """Comprehensive token analysis"""
        try:
            # Extract basic info
            base_token = token_data.get('baseToken', {})
            mint = base_token.get('address')
            
            if not mint or mint in self.scanned_tokens:
                return None
            
            self.stats['tokens_scanned'] += 1
            
            # Get holder analysis
            holder_data = await self.holder_analyzer.analyze_holders(mint)
            
            # Get bonding curve status
            bonding_data = await self.bonding_analyzer.get_bonding_curve_status(
                mint, 
                self._extract_bonding_curve(token_data)
            )
            
            # Create analysis object
            analysis = TokenAnalysis(
                mint=mint,
                symbol=base_token.get('symbol', 'UNKNOWN'),
                name=base_token.get('name', 'Unknown'),
                price=float(token_data.get('priceUsd', 0)),
                market_cap=float(token_data.get('marketCap', 0)),
                liquidity=float(token_data.get('liquidity', {}).get('usd', 0)),
                age_minutes=self._calculate_age(token_data.get('pairCreatedAt', 0)),
                creator=await self._get_creator(mint),
                creation_tx=token_data.get('txns', {}).get('m5', {}).get('buys', 0),
                
                # Bonding curve
                bonding_curve_progress=bonding_data['progress'],
                bonding_curve_address=self._extract_bonding_curve(token_data),
                sol_in_curve=bonding_data['sol_raised'],
                tokens_in_curve=bonding_data['tokens_sold'],
                migration_status=bonding_data['migration_status'],
                
                # Holders
                total_holders=holder_data['total_holders'],
                top_10_percentage=holder_data['top_10_percentage'],
                top_holder_percentage=holder_data['top_holder_percentage'],
                unique_holders=holder_data['unique_holders'],
                bundled_wallets=holder_data['bundled_wallets'],
                dev_holding=holder_data['dev_holding'],
                suspected_insiders=[],
                holder_concentration_score=holder_data['holder_concentration_score'],
                
                # Liquidity
                liquidity_locked=await self._check_liquidity_lock(mint),
                liquidity_lock_duration=0,
                lp_burn_percentage=0,
                pool_creation_time=datetime.now(),
                
                # Technical
                support_levels=[],
                resistance_levels=[],
                current_rsi=50,
                volume_24h=float(token_data.get('volume', {}).get('h24', 0)),
                volume_spike=False,
                price_change_1h=float(token_data.get('priceChange', {}).get('h1', 0)),
                price_change_24h=float(token_data.get('priceChange', {}).get('h24', 0)),
                buy_sell_ratio=self._calculate_buy_sell_ratio(token_data),
                
                # Risk defaults
                risk_level=RiskLevel.MEDIUM,
                rugpull_probability=0,
                bundle_detected=len(holder_data['bundled_wallets']) > 0,
                honeypot_risk=False,
                manipulation_score=0,
                similar_creator_rugs=0,
                
                # AI scoring defaults
                pump_score=0,
                success_probability=0,
                recommended_action="ANALYZING",
                confidence_level=0,
                reasons=[],
                
                # Pattern detection
                wash_trading_detected=False,
                fake_volume_percentage=0,
                bot_activity_score=0,
                organic_growth_score=0
            )
            
            # Perform advanced analysis
            await self._analyze_patterns(analysis)
            await self._analyze_manipulation(analysis)
            await self._analyze_creator_history(analysis)
            
            # Calculate risk scores
            self._calculate_risk_scores(analysis)
            
            # Get AI prediction
            success_prob, confidence = self.ai_analyzer.predict_success(analysis)
            analysis.success_probability = success_prob
            analysis.confidence_level = confidence
            
            # Calculate final pump score
            analysis.pump_score = self._calculate_pump_score(analysis)
            
            # Determine recommendation
            self._determine_recommendation(analysis)
            
            # Store analysis
            self.scanned_tokens[mint] = analysis
            
            # Check if alert worthy
            if analysis.pump_score >= 70 and analysis.risk_level != RiskLevel.RUGPULL:
                self.alert_tokens.append(mint)
                self.stats['opportunities_found'] += 1
                await self._send_alert(analysis)
            elif analysis.risk_level == RiskLevel.RUGPULL:
                self.stats['rugs_detected'] += 1
            
            return analysis
            
        except Exception as e:
            logging.error(f"Token analysis error: {e}")
            return None
    
    async def _analyze_patterns(self, analysis: TokenAnalysis):
        """Detect trading patterns"""
        # Wash trading detection
        analysis.wash_trading_detected = await self._detect_wash_trading(analysis)
        
        # Bot activity scoring
        analysis.bot_activity_score = await self._calculate_bot_activity(analysis)
        
        # Organic growth scoring
        analysis.organic_growth_score = self._calculate_organic_score(analysis)
        
        # Fake volume estimation
        if analysis.wash_trading_detected:
            analysis.fake_volume_percentage = min(
                analysis.bot_activity_score * 100, 
                80
            )
    
    async def _analyze_manipulation(self, analysis: TokenAnalysis):
        """Analyze market manipulation"""
        analysis.manipulation_score = self.ai_analyzer.detect_manipulation_patterns(analysis)
        
        # Additional checks
        if analysis.volume_24h > analysis.liquidity * 10:
            analysis.manipulation_score = min(analysis.manipulation_score + 0.2, 1.0)
            analysis.reasons.append("⚠️ Abnormal volume/liquidity ratio")
    
    async def _analyze_creator_history(self, analysis: TokenAnalysis):
        """Analyze token creator history"""
        creator_analysis = self.ai_analyzer.analyze_creator_history(analysis.creator)
        
        analysis.similar_creator_rugs = creator_analysis.get('rug_rate', 0) * \
                                        creator_analysis.get('total_tokens', 0)
        
        if creator_analysis.get('risk_score', 0) > 0.7:
            analysis.reasons.append(f"🚨 High-risk creator (rug rate: {creator_analysis.get('rug_rate', 0):.1%})")
    
    def _calculate_risk_scores(self, analysis: TokenAnalysis):
        """Calculate comprehensive risk scores"""
        risk_score = 0
        
        # Liquidity risk
        if analysis.liquidity < 5000:
            risk_score += 0.3
            analysis.reasons.append("⚠️ Low liquidity (<$5k)")
        
        # Holder concentration risk
        if analysis.top_10_percentage > 80:
            risk_score += 0.3
            analysis.reasons.append("⚠️ High holder concentration")
        
        # Dev holding risk
        if analysis.dev_holding > 20:
            risk_score += 0.2
            analysis.reasons.append(f"⚠️ High dev holding ({analysis.dev_holding:.1f}%)")
        
        # Bundle risk
        if analysis.bundle_detected:
            risk_score += 0.2
            bundle_count = len(analysis.bundled_wallets)
            analysis.reasons.append(f"⚠️ Bundle detected ({bundle_count} wallets)")
        
        # Age risk
        if analysis.age_minutes < 30:
            risk_score += 0.1
            analysis.reasons.append("⚠️ Very new token")
        
        # Manipulation risk
        risk_score += analysis.manipulation_score * 0.3
        
        # Calculate rugpull probability
        analysis.rugpull_probability = min(risk_score, 1.0)
        
        # Determine risk level
        if analysis.rugpull_probability > 0.8:
            analysis.risk_level = RiskLevel.RUGPULL
        elif analysis.rugpull_probability > 0.6:
            analysis.risk_level = RiskLevel.HIGH
        elif analysis.rugpull_probability > 0.3:
            analysis.risk_level = RiskLevel.MEDIUM
        else:
            analysis.risk_level = RiskLevel.SAFE
    
    def _calculate_pump_score(self, analysis: TokenAnalysis) -> float:
        """Calculate comprehensive pump score"""
        score = 0
        
        # Positive factors
        if analysis.liquidity >= 10000:
            score += 15
            analysis.reasons.append("✅ Strong liquidity")
        
        if analysis.unique_holders >= 200:
            score += 10
            analysis.reasons.append("✅ Good holder count")
        
        if analysis.bonding_curve_progress > 50:
            score += 10
            analysis.reasons.append(f"✅ Bonding curve {analysis.bonding_curve_progress:.1f}% complete")
        
        if analysis.organic_growth_score > 0.7:
            score += 20
            analysis.reasons.append("✅ Organic growth detected")
        
        if analysis.volume_spike and not analysis.wash_trading_detected:
            score += 10
            analysis.reasons.append("📈 Genuine volume spike")
        
        # AI prediction weight
        score += analysis.success_probability * 30
        
        # Negative factors
        score -= analysis.rugpull_probability * 50
        score -= analysis.manipulation_score * 30
        
        # Technical factors
        if 30 < analysis.current_rsi < 70:
            score += 5
        
        # Bonding curve specific
        if analysis.bonding_curve_progress > 80 and analysis.migration_status == "active":
            score += 10
            analysis.reasons.append("🚀 Near migration threshold")
        
        return max(0, min(100, score))
    
    def _determine_recommendation(self, analysis: TokenAnalysis):
        """Determine trading recommendation"""
        if analysis.risk_level == RiskLevel.RUGPULL:
            analysis.recommended_action = "AVOID"
            analysis.reasons.append("🚨 HIGH RUGPULL RISK - AVOID")
        
        elif analysis.pump_score >= 70 and analysis.confidence_level > 0.7:
            analysis.recommended_action = "BUY"
            analysis.reasons.append(f"🟢 Strong buy signal (confidence: {analysis.confidence_level:.1%})")
        
        elif analysis.pump_score >= 50:
            analysis.recommended_action = "WATCH"
            analysis.reasons.append("👀 Monitor for better entry")
        
        else:
            analysis.recommended_action = "PASS"
            analysis.reasons.append("❌ Insufficient positive signals")
    
    async def display_dashboard(self):
        """Display live dashboard"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        with Live(layout, refresh_per_second=1) as live:
            while True:
                # Update header
                header = Panel(
                    f"[bold cyan]🤖 Enhanced Pump.fun Scanner[/bold cyan]\n"
                    f"Tokens Scanned: {self.stats['tokens_scanned']} | "
                    f"Rugs Detected: {self.stats['rugs_detected']} | "
                    f"Opportunities: {self.stats['opportunities_found']}",
                    style="cyan"
                )
                layout["header"].update(header)
                
                # Update main content
                table = self._create_opportunity_table()
                layout["main"].update(Panel(table, title="Top Opportunities"))
                
                # Update footer
                footer = Panel(
                    f"[yellow]Based on research: 98.6% tokens fail | 93% show manipulation[/yellow]",
                    style="yellow"
                )
                layout["footer"].update(footer)
                
                await asyncio.sleep(5)
    
    def _create_opportunity_table(self) -> Table:
        """Create opportunities table"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", width=10)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Price", justify="right", width=12)
        table.add_column("MCap", justify="right", width=10)
        table.add_column("Liq", justify="right", width=10)
        table.add_column("Risk", width=10)
        table.add_column("Action", width=10)
        table.add_column("Bonding %", justify="right", width=10)
        
        # Sort by pump score
        sorted_tokens = sorted(
            self.scanned_tokens.values(),
            key=lambda x: x.pump_score,
            reverse=True
        )[:10]
        
        for token in sorted_tokens:
            risk_color = {
                RiskLevel.SAFE: "green",
                RiskLevel.MEDIUM: "yellow",
                RiskLevel.HIGH: "red",
                RiskLevel.RUGPULL: "red bold"
            }.get(token.risk_level, "white")
            
            action_color = {
                "BUY": "green bold",
                "WATCH": "yellow",
                "PASS": "red",
                "AVOID": "red bold"
            }.get(token.recommended_action, "white")
            
            table.add_row(
                token.symbol,
                f"{token.pump_score:.1f}",
                f"${token.price:.8f}",
                f"${token.market_cap:,.0f}",
                f"${token.liquidity:,.0f}",
                f"[{risk_color}]{token.risk_level.value}[/{risk_color}]",
                f"[{action_color}]{token.recommended_action}[/{action_color}]",
                f"{token.bonding_curve_progress:.1f}%"
            )
        
        return table
    
    async def _send_alert(self, analysis: TokenAnalysis):
        """Send alert for high-score tokens"""
        console.print(f"\n[bold green]🎯 OPPORTUNITY ALERT: {analysis.symbol}[/bold green]")
        console.print(f"Score: {analysis.pump_score:.1f}/100")
        console.print(f"Contract: {analysis.mint}")
        console.print("Reasons:")
        for reason in analysis.reasons[:5]:
            console.print(f"  {reason}")
    
    # Helper methods
    def _calculate_age(self, created_at: int) -> int:
        """Calculate token age in minutes"""
        if created_at:
            return int((time.time() - created_at / 1000) / 60)
        return 0
    
    async def _get_creator(self, mint: str) -> str:
        """Get token creator address"""
        # This would analyze the token creation transaction
        return "unknown"
    
    def _extract_bonding_curve(self, token_data: Dict) -> str:
        """Extract bonding curve address from token data"""
        # This would parse the token data for bonding curve
        return "unknown"
    
    async def _check_liquidity_lock(self, mint: str) -> bool:
        """Check if liquidity is locked"""
        # This would check for LP token burns or locks
        return False
    
    def _calculate_buy_sell_ratio(self, token_data: Dict) -> float:
        """Calculate buy/sell ratio"""
        txns = token_data.get('txns', {}).get('m5', {})
        buys = txns.get('buys', 0)
        sells = txns.get('sells', 0)
        
        if sells == 0:
            return float('inf') if buys > 0 else 1.0
        
        return buys / sells
    
    async def _detect_wash_trading(self, analysis: TokenAnalysis) -> bool:
        """Detect wash trading patterns"""
        # Look for:
        # 1. Repetitive buy/sell patterns
        # 2. Same-size transactions
        # 3. Rapid back-and-forth trades
        
        if analysis.volume_24h > analysis.liquidity * 5:
            return True
        
        return False
    
    async def _calculate_bot_activity(self, analysis: TokenAnalysis) -> float:
        """Calculate bot activity score"""
        score = 0
        
        # Check transaction patterns
        if analysis.buy_sell_ratio > 10 or analysis.buy_sell_ratio < 0.1:
            score += 0.3
        
        # Check for bundle activity
        if len(analysis.bundled_wallets) > 5:
            score += 0.4
        
        # Check volume patterns
        if analysis.volume_spike and analysis.age_minutes < 60:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_organic_score(self, analysis: TokenAnalysis) -> float:
        """Calculate organic growth score"""
        score = 1.0
        
        # Deduct for negative factors
        score -= analysis.bot_activity_score * 0.5
        score -= (1 - min(analysis.unique_holders / 100, 1)) * 0.3
        
        if analysis.wash_trading_detected:
            score -= 0.3
        
        if analysis.bundle_detected:
            score -= 0.2
        
        return max(0, score)
    
    async def _process_new_token(self, log_data: Dict):
        """Process new token creation"""
        # Extract token mint from logs
        # Add to scanning queue
        pass
    
    async def _process_migration(self, log_data: Dict):
        """Process token migration to Raydium"""
        # Extract token mint
        # Update migration status
        self.stats['tokens_migrated'] += 1
    
    async def update_watched_tokens(self):
        """Update watched tokens periodically"""
        while True:
            for mint in list(self.watching_tokens):
                if mint in self.scanned_tokens:
                    # Re-analyze
                    # Check for significant changes
                    pass
            
            await asyncio.sleep(30)