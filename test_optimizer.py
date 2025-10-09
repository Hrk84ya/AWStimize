from iac_optimizer import IaCOptimizer
import json

def analyze_terraform():
    print("🚀 AWS Infrastructure Cost Optimizer")
    print("   Analyzing your Terraform configuration...\n")
    
    optimizer = IaCOptimizer()
    optimizer.train_model()
    
    with open('example_configs/terraform_example.tf', 'r') as f:
        tf_content = f.read()
    
    result = optimizer.analyze_terraform(tf_content)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    summary = result['summary']
    
    # Cost comparison header
    print("=" * 50)
    print("💰 COST ANALYSIS RESULTS")
    print("=" * 50)
    
    current_cost = summary['estimated_monthly_cost']
    optimized_cost = summary['optimized_monthly_cost']
    savings = summary['total_monthly_savings']
    savings_pct = (savings / current_cost * 100) if current_cost > 0 else 0
    
    print(f"Current Monthly Cost:    ${current_cost:>8}")
    print(f"After Optimization:      ${optimized_cost:>8}")
    print(f"Monthly Savings:         ${savings:>8} ({savings_pct:.0f}% reduction)")
    print()
    
    # Infrastructure overview
    print("📈 Infrastructure Overview:")
    print(f"   • {summary['total_resources']} AWS resources")
    print(f"   • {summary['total_dependencies']} dependencies")
    print(f"   • Complexity score: {summary['complexity_score']:.1f}/5.0")
    print()
    
    # Show only resources with costs > 0
    print("💰 Resource Costs:")
    for item in result['cost_breakdown']:
        if item['monthly_cost'] > 0:
            resource_name = item['resource'].replace('aws_', '').replace('_', ' ').title()
            print(f"   • {resource_name}: ${item['monthly_cost']}")
            
            if item['rightsizing']:
                rs = item['rightsizing']
                print(f"     💡 Optimize: Switch to {rs['suggested_type']} → Save ${rs['monthly_savings']:.2f}/month")
    print()
    
    # Optimization summary
    opts = result['optimizations']
    cost_opts = len(opts['cost_optimizations'])
    
    if cost_opts > 0:
        print("🚀 RECOMMENDED ACTIONS:")
        print("-" * 30)
        for i, opt in enumerate(opts['cost_optimizations'], 1):
            resource_name = opt['resource_name'].replace('aws_', '').replace('_', ' ').title()
            print(f"{i}. {resource_name}")
            print(f"   Action: {opt['recommendation']}")
            if 'potential_savings' in opt:
                print(f"   Impact: {opt['potential_savings']}")
            print()
    else:
        print("✅ Your infrastructure is already cost-optimized!")
    
    # Bottom line summary
    if savings > 0:
        annual_savings = savings * 12
        print("=" * 50)
        print(f"🎆 POTENTIAL ANNUAL SAVINGS: ${annual_savings:.2f}")
        print("=" * 50)

if __name__ == "__main__":
    analyze_terraform()