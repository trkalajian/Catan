from colored import Fore, Back, Style

Fore.red
'\x1b[38;5;1m'

Back.red
'\x1b[48;5;1m'

Style.reset
'\x1b[0m'

Fore.rgb('100%', '50%', '30%')
'\x1b[38;2;255;130;79m'

print(f'{Fore.white}{Back.green}Colored is Awesome!!!{Style.reset}')