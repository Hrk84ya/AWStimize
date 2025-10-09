# 🚀 AWS Infrastructure Cost Optimizer

An AI-powered tool that analyzes your Terraform configurations and provides **real cost savings recommendations** using Graph Neural Networks. Get instant insights into your AWS spending and optimize your infrastructure costs.

## 💰 What It Does

- **Analyzes AWS costs** with real pricing data (EC2, RDS, S3, etc.)
- **Suggests rightsizing** for oversized instances
- **Calculates exact savings** in dollars and percentages
- **Shows before/after costs** for easy comparison
- **Uses AI** to learn resource relationships and patterns

## 🎯 Key Benefits

✅ **Instant cost analysis** - Know your monthly AWS spend  
✅ **Real savings recommendations** - Get specific instance downsizing suggestions  
✅ **ROI calculations** - See exact dollar savings per optimization  
✅ **Easy to use** - Just put your Terraform file and run  
✅ **Production ready** - Based on actual AWS pricing data  

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Terraform Configuration
Put your `.tf` file in `example_configs/terraform_example.tf`

### 3. Run Analysis
```bash
python test_optimizer.py
```

### 4. Get Results
```
🚀 AWS Infrastructure Cost Optimizer
==================================================
💰 COST ANALYSIS RESULTS
==================================================
Current Monthly Cost:    $149.17
After Optimization:      $ 75.58
Monthly Savings:         $ 73.59 (49% reduction)

🚀 RECOMMENDED ACTIONS:
1. Instance Unused
   Action: Rightsize from t3.medium to t3.small
   Impact: $15.19/month (50.0%)

2. Rds Instance Database
   Action: Rightsize from db.t3.large to db.t3.medium
   Impact: $58.40/month (50.0%)

==================================================
🎆 POTENTIAL ANNUAL SAVINGS: $883.08
==================================================
```

## 📁 Project Structure

```
terra_testing/
├── iac_parser.py          # Terraform/CloudFormation parser
├── gnn_model.py           # Graph Neural Network model
├── aws_pricing.py         # Real AWS pricing data
├── iac_optimizer.py       # Main optimization engine
├── test_optimizer.py      # User-friendly analysis tool
├── requirements.txt       # Dependencies
└── example_configs/
    └── terraform_example.tf  # Your Terraform file goes here
```

## 💡 How It Works

1. **Parses** your Terraform configuration
2. **Builds** a dependency graph of your resources
3. **Calculates** real AWS costs using current pricing
4. **Applies** Graph Neural Network to learn optimization patterns
5. **Suggests** specific rightsizing recommendations
6. **Shows** exact cost savings and ROI

## 🎯 Supported Resources

- **EC2 Instances**: t2.nano to t3.2xlarge with real pricing
- **RDS Databases**: All t3 instance classes + storage costs
- **S3 Buckets**: Basic storage pricing
- **Network Resources**: VPC, subnets, security groups (free tier)

## 📊 Real Example Results

**Before Optimization**: $149.17/month  
**After Optimization**: $75.58/month  
**Monthly Savings**: $73.59 (49% reduction)  
**Annual Savings**: $883.08  

## 🎉 Perfect For

- **DevOps Engineers** optimizing AWS costs
- **Cloud Architects** reviewing infrastructure efficiency  
- **Finance Teams** tracking cloud spending
- **Startups** maximizing their AWS credits
- **Enterprises** reducing cloud waste

---

**Ready to save money on AWS?** Put your Terraform file in `example_configs/terraform_example.tf` and run `python test_optimizer.py` to see your potential savings!
