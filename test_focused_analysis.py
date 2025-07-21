#!/usr/bin/env python3
"""Test script to verify the focused brand extraction functionality.
"""

import asyncio
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.services.llm_analysis import LLMAnalysisService


async def test_focused_analysis():
    """Test the focused brand extraction."""
    service = LLMAnalysisService()

    # Test case 1: Car brands (should only extract car brands, not tire brands)
    car_responses = {
        'claude': 'The best cars are Toyota Camry, Honda Civic, and Ford Mustang. Michelin tires are also good for these cars.',
        'gemini': 'I recommend BMW X5, Mercedes C-Class, and Audi A4. Goodyear tires work well with these vehicles.',
        'gpt': 'Top car brands include Tesla Model 3, Porsche 911, and Ferrari F8. Bridgestone tires are excellent.',
    }

    print('🧪 Testing Focused Brand Extraction...')
    print('📝 Test Case 1: Car brands (should exclude tire brands)')
    print()

    try:
        # Test car brands analysis
        print('🔍 Analyzing car responses...')
        car_analysis = await service.analyze_responses(car_responses)

        print('📊 Car Analysis Results:')
        for model, brands in car_analysis.items():
            print(f'\n  🤖 {model.upper()}:')
            if brands:
                for brand in brands:
                    print(
                        f'    - {brand["brand"]} (pos: {brand["position"]}, sentiment: {brand["sentiment"]}, links: {brand["contains_link"]})'
                    )
            else:
                print('    - No brands detected')

        print()

        # Test case 2: Restaurant brands (should only extract restaurant names)
        restaurant_responses = {
            'claude': "Best restaurants are McDonald's, Burger King, and KFC. Uber Eats delivers to all of them.",
            'gemini': "I recommend Subway, Domino's Pizza, and Pizza Hut. DoorDash is a great delivery service.",
            'gpt': "Top restaurants include Chipotle, Taco Bell, and Wendy's. Grubhub offers delivery for these places.",
        }

        print('📝 Test Case 2: Restaurant brands (should exclude delivery services)')
        print()

        # Test restaurant brands analysis
        print('🔍 Analyzing restaurant responses...')
        restaurant_analysis = await service.analyze_responses(restaurant_responses)

        print('📊 Restaurant Analysis Results:')
        for model, brands in restaurant_analysis.items():
            print(f'\n  🤖 {model.upper()}:')
            if brands:
                for brand in brands:
                    print(
                        f'    - {brand["brand"]} (pos: {brand["position"]}, sentiment: {brand["sentiment"]}, links: {brand["contains_link"]})'
                    )
            else:
                print('    - No brands detected')

        print()
        print('✅ Focused analysis test completed successfully!')

    except Exception as e:
        print(f'❌ Error during testing: {str(e)}')


if __name__ == '__main__':
    asyncio.run(test_focused_analysis())
