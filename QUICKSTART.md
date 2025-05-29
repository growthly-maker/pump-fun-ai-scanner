# 🚀 Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- Solana wallet with some SOL for transactions (optional, only for trading)
- Basic understanding of cryptocurrency trading risks

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/growthly-maker/pump-fun-ai-scanner.git
   cd pump-fun-ai-scanner
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Basic Usage

### 1. Run in Interactive Mode (Recommended for beginners)
```bash
python scanner.py
```

This will start the scanner in interactive mode where you can:
- Start/stop scanning
- Analyze specific tokens
- View statistics
- Export data
- Update configuration

### 2. Run in Scan Mode
```bash
python scanner.py --mode scan
```

This starts continuous scanning immediately.

### 3. Use a Preset Configuration
```bash
# Conservative (safer, fewer alerts)
python scanner.py --preset conservative

# Moderate (balanced approach)
python scanner.py --preset moderate

# Aggressive (more alerts, higher risk)
python scanner.py --preset aggressive
```

## Understanding the Output

### Dashboard View
```
🤖 Enhanced Pump.fun Scanner
Tokens Scanned: 245 | Rugs Detected: 223 | Opportunities: 3

╭─────────────────── Top Opportunities ────────────────────╮
│ Symbol   Score  Price        MCap      Liq      Risk  Action   Bonding % │
│ PEPE     85.3   $0.00000123  $45.2K    $12.3K   SAFE  BUY      67.8%    │
│ DOGE2    72.1   $0.00000456  $23.1K    $8.5K    MED   WATCH    45.2%    │
│ MOON     68.9   $0.00000789  $67.8K    $15.6K   MED   WATCH    89.1%    │
╰──────────────────────────────────────────────────────────────────────────╯

Based on research: 98.6% tokens fail | 93% show manipulation
```

### Alert Example
```
🎯 OPPORTUNITY ALERT: PEPE
Score: 85.3/100
Contract: 2CAETyBJk83aGsmU65uLP3s78pigdUNJv3uRWamFpump
Reasons:
  ✅ Strong liquidity
  ✅ Good holder count
  ✅ Organic growth detected
```

## Key Metrics Explained

### Pump Score (0-100)
- **70-100**: Strong opportunity, consider buying
- **50-70**: Worth monitoring, wait for better entry
- **30-50**: Caution advised
- **0-30**: Avoid

### Risk Levels
- **SAFE**: Low risk indicators
- **MEDIUM**: Normal risk, monitor closely
- **HIGH**: Elevated risk, be very careful
- **RUGPULL**: Extreme risk, avoid completely

### Bonding Curve Progress
- Shows % completion toward $68k market cap
- At 100%, token migrates to Raydium
- Higher % = closer to guaranteed liquidity

## Configuration Options

### Essential Settings
```env
# Minimum liquidity in USD
MIN_LIQUIDITY_USD=5000

# Minimum number of holders
MIN_HOLDERS=100

# Maximum developer holding percentage
MAX_DEV_HOLDING=15

# Pump score threshold for alerts
PUMP_SCORE_THRESHOLD=70
```

### Alert Configuration
```env
# Discord alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Common Commands

### Export Data
In interactive mode, choose "export" to save results:
- CSV format: Best for Excel analysis
- JSON format: Best for programmatic access

### Analyze Specific Token
```
Enter token mint address: 2CAETyBJk83aGsmU65uLP3s78pigdUNJv3uRWamFpump
```

### View Statistics
Shows:
- Total tokens scanned
- Rugpulls detected (usually ~98%)
- Opportunities found
- Tokens that completed bonding curve

## Tips for Success

1. **Start Conservative**: Use conservative preset initially
2. **Monitor Bonding Progress**: Tokens near 100% are about to get guaranteed liquidity
3. **Check Multiple Factors**: Don't rely on score alone
4. **Verify Liquidity**: Ensure liquidity is locked or burned
5. **Watch Dev Holdings**: <10% is ideal, >20% is risky

## Safety Guidelines

⚠️ **IMPORTANT WARNINGS**:
- 98.6% of Pump.fun tokens fail
- 93% show signs of manipulation
- Never invest more than you can afford to lose
- This tool is for research only, not financial advice

## Troubleshooting

### "Failed to connect to RPC"
- Check your internet connection
- Try a different RPC URL
- Consider using a paid RPC service

### "No opportunities found"
- This is normal - legitimate opportunities are rare
- Lower your threshold settings if needed
- Remember: no opportunities is better than bad ones

### High CPU/Memory Usage
- Reduce concurrent scans in config
- Increase scan interval
- Use a more powerful machine

## Advanced Usage

### Custom RPC Endpoint
```env
SOLANA_RPC_URL=https://your-rpc-endpoint.com
```

### Enable Auto-Trading (USE WITH EXTREME CAUTION)
```env
ENABLE_AUTO_TRADING=true
PRIVATE_KEY=your_wallet_private_key
DEFAULT_BUY_AMOUNT=0.1
```

### Run with Debug Logging
```bash
python scanner.py --debug
```

## Next Steps

1. Read the full README for detailed information
2. Join our Discord/Telegram for updates
3. Contribute improvements via GitHub
4. Share your success stories (and failures!)

## Support

- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share strategies
- Pull Requests: Contribute improvements

Remember: In the world of memecoins, patience and caution are your best friends! 🚀