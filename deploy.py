#!/usr/bin/env python3
"""
Deployment helper script for Heroku
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_requirements():
    """Check if required tools are installed"""
    print("🔍 Checking requirements...")
    
    # Check if git is available
    if not run_command("git --version", "Checking Git"):
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Check if heroku CLI is available
    if not run_command("heroku --version", "Checking Heroku CLI"):
        print("❌ Heroku CLI is not installed. Please install it from https://devcenter.heroku.com/articles/heroku-cli")
        return False
    
    # Check if model file exists
    if not os.path.exists("models/final_model.pkl"):
        print("❌ Model file not found. Please run 'python main.py' first to train the model.")
        return False
    
    print("✅ All requirements met!")
    return True

def deploy_to_heroku():
    """Deploy the application to Heroku"""
    print("🚀 Starting Heroku deployment...")
    
    # Check if we're in a git repository
    if not os.path.exists(".git"):
        print("📁 Initializing Git repository...")
        if not run_command("git init", "Initializing Git"):
            return False
    
    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return False
    
    # Commit changes
    if not run_command('git commit -m "Deploy Disease Diagnosis API"', "Committing changes"):
        return False
    
    # Check if heroku remote exists
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if "heroku" not in result.stdout:
        print("🔗 Please create a Heroku app first:")
        print("   heroku create your-app-name")
        print("   Then run this script again.")
        return False
    
    # Deploy to Heroku
    if not run_command("git push heroku main", "Deploying to Heroku"):
        # Try with master branch if main fails
        print("🔄 Trying with master branch...")
        if not run_command("git push heroku master", "Deploying to Heroku"):
            return False
    
    print("🎉 Deployment completed successfully!")
    print("🌐 Your API should be available at: https://your-app-name.herokuapp.com")
    print("📖 API Documentation: https://your-app-name.herokuapp.com/docs")
    
    return True

def main():
    """Main deployment function"""
    print("🏥 Disease Diagnosis API - Heroku Deployment Helper")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Ask for confirmation
    print("\n⚠️  Before deploying, make sure you have:")
    print("   1. Created a Heroku app (heroku create your-app-name)")
    print("   2. Logged in to Heroku (heroku login)")
    print("   3. Committed your model file to Git")
    
    response = input("\n🤔 Do you want to continue with deployment? (y/N): ")
    if response.lower() != 'y':
        print("❌ Deployment cancelled.")
        sys.exit(0)
    
    # Deploy
    if deploy_to_heroku():
        print("\n🎉 Deployment successful!")
        print("\n📋 Next steps:")
        print("   1. Check your app: heroku open")
        print("   2. View logs: heroku logs --tail")
        print("   3. Test the API: heroku run python test_api.py")
    else:
        print("\n❌ Deployment failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
