# 🚀 AWS Infrastructure Cost Optimizer

An AI-powered tool that analyzes your Terraform configurations and provides **real cost savings recommendations** using Graph Neural Networks. Get instant insights into your AWS spending and optimize your infrastructure costs.

## 💰 What It Does

- **Analyzes AWS costs** with real pricing data (EC2, RDS, S3, etc.)
- **Suggests rightsizing** for oversized instances
- **Detects security vulnerabilities** (open security groups, unencrypted storage)
- **Calculates exact savings** in dollars and percentages
- **Uses ML model** trained on realistic infrastructure patterns
- **Shows before/after costs** for easy comparison

## 🎯 Key Benefits

✅ **Instant analysis** - Know your monthly AWS spend and security posture  
✅ **Real recommendations** - Get specific rightsizing and security improvements  
✅ **ROI calculations** - See exact dollar savings per optimization  
✅ **Security scanning** - Detect misconfigurations and vulnerabilities  
✅ **Easy to use** - Just put your Terraform file and run  
✅ **Production ready** - Based on actual AWS pricing and security best practices  

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
Current Monthly Cost:    $270.64
After Optimization:      $136.32
Monthly Savings:         $134.32 (50% reduction)

🚀 RECOMMENDED ACTIONS:
1. Instance.Web
   Action: Rightsize from t3.xlarge to t3.large
   Impact: $60.73/month (50.0%)

2. Instance.Unused
   Action: Rightsize from t3.medium to t3.small
   Impact: $15.19/month (50.0%)

🔒 SECURITY IMPROVEMENTS:
1. Security Group.Web
   Issue: Open to all IPs (0.0.0.0/0)
   Action: Restrict overly permissive security group rules
   Severity: High

==================================================
🎆 POTENTIAL ANNUAL SAVINGS: $1,611.84
==================================================
```

## 📁 Project Structure

```
AWStimize/
├── iac_parser.py              # Terraform/CloudFormation parser
├── gnn_model.py               # Graph Neural Network model
├── aws_pricing.py             # Real AWS pricing data
├── iac_optimizer.py           # Main optimization engine
├── test_optimizer.py          # User-friendly analysis tool
├── synthetic_data_generator.py # ML training data generator
├── requirements.txt           # Dependencies
└── example_configs/
    └── terraform_example.tf      # Your Terraform file goes here
```

## 💡 How It Works

1. **Parses** your Terraform configuration
2. **Builds** a dependency graph of your resources
3. **Calculates** real AWS costs using current pricing
4. **Applies** ML model trained on realistic usage patterns
5. **Detects** security vulnerabilities and misconfigurations
6. **Suggests** specific rightsizing and security improvements

## 🎯 Supported Resources

- **EC2 Instances**: t2.nano to t3.2xlarge with real pricing
- **RDS Databases**: All t3 instance classes + storage costs
- **S3 Buckets**: Basic storage pricing
- **Security Groups**: Open rule detection and recommendations
- **Network Resources**: VPC, subnets, security groups

## 📊 Real Example Results

**Before Optimization**: $270.64/month  
**After Optimization**: $136.32/month  
**Monthly Savings**: $134.32 (50% reduction)  
**Annual Savings**: $1,611.84  
**Security Issues Found**: 3 (open security groups, unencrypted storage)  

## 🎉 Perfect For

- **DevOps Engineers** optimizing AWS costs and security
- **Cloud Architects** reviewing infrastructure efficiency  
- **Security Teams** identifying misconfigurations
- **Finance Teams** tracking cloud spending
- **Startups** maximizing their AWS credits
- **Enterprises** reducing cloud waste and security risks

---

**Ready to save money on AWS?** Put your Terraform file in `example_configs/terraform_example.tf` and run `python test_optimizer.py` to see your potential savings!
