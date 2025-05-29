#!/usr/bin/env python3
"""
Main entry point for the Enhanced Pump.fun Scanner
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
import base58
import argparse
import logging
from rich.console import Console
from rich.prompt import Prompt, Confirm
import signal

# Import our modules
from enhanced_scanner_main import EnhancedPumpScanner
from config import Config, load_config

# Load environment variables
load_dotenv()

console = Console()

class PumpScannerApp:
    """Main application class"""
    
    def __init__(self):
        self.config = load_config()
        self.scanner = None
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scanner.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        console.print("\n[yellow]Shutting down scanner...[/yellow]")
        self.running = False
        sys.exit(0)
    
    async def initialize(self):
        """Initialize scanner components"""
        console.print("[cyan]Initializing scanner...[/cyan]")
        
        # Setup RPC client
        rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
        self.rpc_client = AsyncClient(rpc_url)
        
        # Test connection
        try:
            version = await self.rpc_client.get_version()
            console.print(f"[green]✓ Connected to Solana RPC (version: {version.value.solana_core})[/green]")
        except Exception as e:
            console.print(f"[red]Failed to connect to RPC: {e}[/red]")
            return False
        
        # Initialize scanner
        self.scanner = EnhancedPumpScanner(self.rpc_client)
        
        return True
    
    async def run_scanner(self):
        """Run the scanner"""
        self.running = True
        
        console.print("\n[bold cyan]🚀 Enhanced Pump.fun Scanner[/bold cyan]")
        console.print("[yellow]Detecting rugpulls and finding opportunities...[/yellow]\n")
        
        # Display config
        self._display_config()
        
        # Start scanning
        await self.scanner.start_scanning()
    
    def _display_config(self):
        """Display current configuration"""
        console.print("[cyan]Configuration:[/cyan]")
        console.print(f"  • Min Liquidity: ${self.config.min_liquidity:,}")
        console.print(f"  • Min Holders: {self.config.min_holders}")
        console.print(f"  • Max Dev Holding: {self.config.max_dev_holding}%")
        console.print(f"  • Pump Score Threshold: {self.config.pump_score_threshold}")
        console.print(f"  • Auto Trading: {'Enabled' if self.config.enable_auto_trading else 'Disabled'}")
        console.print("")
    
    async def run_interactive(self):
        """Run in interactive mode"""
        console.print("[bold cyan]Interactive Mode[/bold cyan]\n")
        
        while True:
            choice = Prompt.ask(
                "\nWhat would you like to do?",
                choices=["scan", "analyze", "config", "stats", "export", "quit"]
            )
            
            if choice == "scan":
                await self.run_scanner()
            elif choice == "analyze":
                await self.analyze_specific_token()
            elif choice == "config":
                await self.update_config()
            elif choice == "stats":
                self.display_stats()
            elif choice == "export":
                await self.export_data()
            elif choice == "quit":
                break
    
    async def analyze_specific_token(self):
        """Analyze a specific token"""
        mint = Prompt.ask("Enter token mint address")
        
        console.print(f"\n[cyan]Analyzing token {mint}...[/cyan]")
        
        # Fetch token data from DexScreener
        # Run analysis
        # Display results
        
    def display_stats(self):
        """Display scanner statistics"""
        if not self.scanner:
            console.print("[yellow]Scanner not initialized[/yellow]")
            return
        
        stats = self.scanner.stats
        
        console.print("\n[bold cyan]Scanner Statistics[/bold cyan]")
        console.print(f"Tokens Scanned: {stats['tokens_scanned']}")
        console.print(f"Rugs Detected: {stats['rugs_detected']} ({stats['rugs_detected']/max(stats['tokens_scanned'], 1)*100:.1f}%)")
        console.print(f"Opportunities Found: {stats['opportunities_found']}")
        console.print(f"Tokens Migrated: {stats['tokens_migrated']}")
    
    async def export_data(self):
        """Export scanned data"""
        if not self.scanner or not self.scanner.scanned_tokens:
            console.print("[yellow]No data to export[/yellow]")
            return
        
        filename = Prompt.ask("Export filename", default="pump_scan_results.csv")
        
        # Convert to DataFrame and export
        import pandas as pd
        
        data = []
        for token in self.scanner.scanned_tokens.values():
            data.append({
                'symbol': token.symbol,
                'mint': token.mint,
                'pump_score': token.pump_score,
                'risk_level': token.risk_level.value,
                'price': token.price,
                'market_cap': token.market_cap,
                'liquidity': token.liquidity,
                'holders': token.total_holders,
                'dev_holding': token.dev_holding,
                'bonding_progress': token.bonding_curve_progress,
                'recommendation': token.recommended_action
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        console.print(f"[green]✓ Exported {len(data)} tokens to {filename}[/green]")
    
    async def update_config(self):
        """Update configuration interactively"""
        console.print("\n[cyan]Update Configuration[/cyan]")
        
        # Update each setting
        self.config.min_liquidity = int(Prompt.ask(
            "Minimum liquidity (USD)", 
            default=str(self.config.min_liquidity)
        ))
        
        self.config.min_holders = int(Prompt.ask(
            "Minimum holders", 
            default=str(self.config.min_holders)
        ))
        
        self.config.max_dev_holding = float(Prompt.ask(
            "Maximum dev holding %", 
            default=str(self.config.max_dev_holding)
        ))
        
        # Save config
        self.config.save()
        console.print("[green]✓ Configuration updated[/green]")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Pump.fun Scanner')
    parser.add_argument('--mode', choices=['scan', 'interactive'], 
                       default='interactive', help='Running mode')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create app instance
    app = PumpScannerApp()
    
    # Initialize
    if not await app.initialize():
        console.print("[red]Failed to initialize scanner[/red]")
        return
    
    try:
        if args.mode == 'scan':
            await app.run_scanner()
        else:
            await app.run_interactive()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scanner stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logging.exception("Scanner error")
    finally:
        # Cleanup
        if app.rpc_client:
            await app.rpc_client.close()

if __name__ == "__main__":
    asyncio.run(main())