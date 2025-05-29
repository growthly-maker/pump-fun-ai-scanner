# 🤖 Pump.fun AI Scanner - Advanced Token Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Solana](https://img.shields.io/badge/Solana-Mainnet-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Overview

An AI-powered scanner for Pump.fun tokens on Solana with advanced rugpull detection, pump probability analysis, and real-time monitoring. Based on research showing that **98.6% of Pump.fun tokens fail** and **93% show manipulation signs**, this scanner helps identify the rare legitimate opportunities.

## 📊 Key Statistics
- Only **1.21%** of Pump.fun tokens reach DEX listing threshold
- **98%** of tokens show signs of manipulation
- **93%** of Raydium pools exhibit soft rug pull characteristics
- Bonding curve completion requires ~$68,000-$73,000 market cap

## 🎯 Features

### Core Analysis
- **AI-Powered Pump Score**: Machine learning model that adapts based on outcomes
- **Rugpull Detection**: Multi-factor analysis including:
  - Liquidity lock status
  - Developer holdings analysis
  - Bundle wallet detection
  - Holder distribution patterns
  - Token age and creation patterns

### Real-Time Monitoring
- DexScreener integration for live data
- Bonding curve progress tracking
- Migration to Raydium detection
- Volume spike alerts
- Support/Resistance level calculation

### Risk Assessment
- 4-tier risk classification (Safe, Medium, High, Rugpull)
- Rugpull probability calculation
- Bundle detection algorithm
- Honeypot risk analysis

### Smart Features
- Learning from trading outcomes
- Pattern recognition database
- Similar token performance analysis
- Automated recommendations (BUY/SELL/WAIT)

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/growthly-maker/pump-fun-ai-scanner.git
cd pump-fun-ai-scanner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

```txt
aiohttp>=3.8.0
solana>=0.30.0
solders>=0.20.0
numpy>=1.24.0
pandas>=2.0.0
colorama>=0.4.6
```

## 🚀 Quick Start

```python
# Run the scanner
python scanner.py

# Or use the enhanced version with full features
python enhanced_scanner.py
```

## 💡 Usage

### Basic Scanning
```python
from scanner import PumpScanner
import asyncio

async def main():
    scanner = PumpScanner()
    await scanner.start_scanning()

asyncio.run(main())
```

### With Trading Bot Integration
```python
from enhanced_scanner import EnhancedPumpBot
from config import BotConfig

bot = EnhancedPumpBot(private_key, BotConfig())
await bot.run_smart_scanner()
```

## 📊 Understanding the Scores

### Pump Score (0-100)
- **70-100**: Strong buy signal
- **50-70**: Monitor closely
- **30-50**: Caution advised
- **0-30**: Avoid/Sell signal

### Risk Levels
- **SAFE**: Low risk, good fundamentals
- **MEDIUM**: Normal risk, monitor closely
- **HIGH**: Elevated risk, trade carefully
- **RUGPULL**: Extreme risk, avoid

## 🔍 Detection Methods

### 1. Holder Analysis
- Top 10 holders percentage
- Unique holder count
- Bundle wallet detection
- Developer holdings tracking

### 2. Liquidity Analysis
- Minimum $5,000 liquidity threshold
- Volume/liquidity ratio monitoring
- Liquidity lock verification

### 3. Technical Analysis
- RSI calculation
- Support/resistance levels
- Price change tracking (1h, 24h)
- Volume spike detection

### 4. AI Pattern Recognition
- Historical performance correlation
- Similar token analysis
- Creator pattern tracking
- Market timing analysis

## ⚠️ Risk Warnings

1. **High Failure Rate**: 98.6% of Pump.fun tokens fail to complete bonding curve
2. **Manipulation Risk**: 93% of pools show manipulation characteristics
3. **Fast Market**: Tokens can pump and dump within minutes
4. **Bundle Risk**: Many tokens use bundled wallets to fake activity

## 🛡️ Safety Features

- Automatic rugpull detection
- Bundle wallet identification
- Suspicious name filtering
- Age-based risk assessment
- Liquidity verification

## 📈 Bonding Curve Mechanics

- Initial supply: 1 billion tokens
- 793 million allocated to bonding curve
- Completion at ~$68,000-$73,000 market cap
- Requires 85 SOL collected
- Automatic migration to Raydium/PumpSwap

## 🔧 Configuration

Edit `config.py` to customize:
```python
class Config:
    # Risk thresholds
    MIN_LIQUIDITY = 5000  # USD
    MAX_DEV_HOLDING = 15  # Percentage
    MIN_HOLDERS = 100
    
    # Trading parameters
    DEFAULT_BUY_AMOUNT = 0.1  # SOL
    SLIPPAGE = 10  # Percentage
    
    # AI weights (auto-adjusted)
    INITIAL_WEIGHTS = {
        'liquidity': 0.15,
        'holders_distribution': 0.20,
        'volume_trend': 0.15,
        'dev_holding': -0.25,
        'bundle_penalty': -0.30
    }
```

## 📊 Data Sources

- **DexScreener API**: Real-time price and volume data
- **Solana RPC**: On-chain transaction analysis
- **Pump.fun Events**: Token creation monitoring
- **Raydium**: Migration tracking

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📜 License

MIT License - see LICENSE file for details

## ⚡ Performance Tips

1. Use a paid RPC endpoint for better reliability
2. Adjust cooldown periods based on your RPC limits
3. Monitor multiple tokens in parallel for efficiency
4. Keep the AI model weights file backed up

## 🔮 Future Enhancements

- [ ] Social sentiment analysis integration
- [ ] Telegram/Discord alert bot
- [ ] Web dashboard interface
- [ ] Advanced backtesting framework
- [ ] Multi-chain support (Base, Blast)

## 📞 Support

- Issues: [GitHub Issues](https://github.com/growthly-maker/pump-fun-ai-scanner/issues)
- Discussions: [GitHub Discussions](https://github.com/growthly-maker/pump-fun-ai-scanner/discussions)

## ⚖️ Disclaimer

This tool is for educational purposes only. Cryptocurrency trading carries high risk. Always do your own research and never invest more than you can afford to lose. The developers are not responsible for any financial losses.

---

**Remember**: In the world of memecoins, if something seems too good to be true, it probably is. Stay safe and trade responsibly! 🚀