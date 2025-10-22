from iac_optimizer import IaCOptimizer
import json

def analyze_terraform():
    print("🚀 AWS Infrastructure Cost Optimizer")
    print("   Analyzing your configuration...\n")
    
    optimizer = IaCOptimizer()
    print("Using pre-trained optimization patterns...")
    print()
    
    # Try comprehensive example first, fall back to simple example
    try:
        with open('example_configs/comprehensive_terraform.tf', 'r') as f:
            tf_content = f.read()
        print("📊 Analyzing comprehensive AWS infrastructure...")
    except FileNotFoundError:
        with open('example_configs/terraform_example.tf', 'r') as f:
            tf_content = f.read()
        print("📊 Analyzing basic AWS infrastructure...")
    
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
    
    # Group resources by category and show costs
    print("💰 Resource Costs by Category:")
    
    # Group resources by service type
    service_costs = {}
    for item in result['cost_breakdown']:
        if item['monthly_cost'] > 0:
            service_type = item['type'].replace('aws_', '').split('_')[0].title()
            if service_type not in service_costs:
                service_costs[service_type] = []
            service_costs[service_type].append(item)
    
    # Display grouped costs
    for service_type, resources in service_costs.items():
        total_service_cost = sum(r['monthly_cost'] for r in resources)
        print(f"\n   📦 {service_type} Services: ${total_service_cost:.2f}/month")
        
        for item in resources:
            resource_name = item['resource'].split('.')[-1].replace('_', '-')
            print(f"      • {resource_name}: ${item['monthly_cost']:.2f}")
            
            if item['rightsizing']:
                rs = item['rightsizing']
                print(f"        💡 Optimize: {rs['suggested_type']} → Save ${rs['monthly_savings']:.2f}/month")
    print()
    
    # Optimization summary
    opts = result['optimizations']
    cost_opts = opts['cost_optimizations']
    security_opts = opts['security_improvements']
    
    if cost_opts:
        print("🚀 RECOMMENDED ACTIONS:")
        print("-" * 30)
        for i, opt in enumerate(cost_opts, 1):
            resource_name = opt['resource_name'].replace('aws_', '').replace('_', ' ').title()
            print(f"{i}. {resource_name}")
            print(f"   Action: {opt['recommendation']}")
            if 'potential_savings' in opt:
                print(f"   Impact: {opt['potential_savings']}")
            elif 'ml_confidence' in opt:
                print(f"   ML Confidence: {opt['ml_confidence']}")
            print()
    
    if security_opts:
        print("🔒 SECURITY IMPROVEMENTS:")
        print("-" * 30)
        for i, opt in enumerate(security_opts, 1):
            resource_name = opt['resource_name'].replace('aws_', '').replace('_', ' ').title()
            print(f"{i}. {resource_name}")
            print(f"   Issue: {opt['issue']}")
            print(f"   Action: {opt['recommendation']}")
            print(f"   Severity: {opt['severity'].title()}")
            print()
    
    if not cost_opts and not security_opts:
        print("✅ Your infrastructure appears well-optimized!")
    
    # Bottom line summary
    if savings > 0:
        annual_savings = savings * 12
        print("=" * 50)
        print(f"🎆 POTENTIAL ANNUAL SAVINGS: ${annual_savings:.2f}")
        print("=" * 50)
    
    if security_opts:
        print(f"🔒 Found {len(security_opts)} security issues to address")

if __name__ == "__main__":
    analyze_terraform()