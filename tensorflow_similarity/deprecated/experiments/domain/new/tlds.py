# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TLDS = [
    'aaa', 'aarp', 'abb', 'abc', 'able', 'ac', 'aco', 'ad', 'adac', 'ads',
    'ae', 'aeg', 'aero', 'af', 'afl', 'ag', 'ai', 'aig', 'aigo', 'akdn', 'al',
    'ally', 'am', 'amex', 'anz', 'ao', 'aol', 'app', 'aq', 'ar', 'arab',
    'army', 'arpa', 'art', 'arte', 'as', 'asda', 'asia', 'at', 'au', 'audi',
    'auto', 'aw', 'aws', 'ax', 'axa', 'az', 'ba', 'baby', 'band', 'bank',
    'bar', 'bb', 'bbc', 'bbt', 'bbva', 'bcg', 'bcn', 'bd', 'be', 'beer',
    'best', 'bet', 'bf', 'bg', 'bh', 'bi', 'bid', 'bike', 'bing', 'bio', 'biz',
    'bj', 'blog', 'blue', 'bm', 'bms', 'bmw', 'bn', 'bnl', 'bo', 'bofa', 'bom',
    'bond', 'boo', 'book', 'bot', 'box', 'br', 'bs', 'bt', 'buy', 'buzz', 'bv',
    'bw', 'by', 'bz', 'bzh', 'ca', 'cab', 'cafe', 'cal', 'call', 'cam', 'camp',
    'car', 'care', 'cars', 'casa', 'case', 'cash', 'cat', 'cba', 'cbn', 'cbre',
    'cbs', 'cc', 'cd', 'ceb', 'ceo', 'cern', 'cf', 'cfa', 'cfd', 'cg', 'ch',
    'chat', 'ci', 'citi', 'city', 'ck', 'cl', 'club', 'cm', 'cn', 'co', 'com',
    'cool', 'coop', 'cr', 'crs', 'csc', 'cu', 'cv', 'cw', 'cx', 'cy', 'cyou',
    'cz', 'dad', 'data', 'date', 'day', 'dclk', 'dds', 'de', 'deal', 'dell',
    'desi', 'dev', 'dhl', 'diet', 'dish', 'diy', 'dj', 'dk', 'dm', 'dnp', 'do',
    'docs', 'dog', 'doha', 'dot', 'dtv', 'duck', 'duns', 'dvag', 'dvr', 'dz',
    'eat', 'ec', 'eco', 'edu', 'ee', 'eg', 'er', 'erni', 'es', 'esq', 'et',
    'eu', 'eus', 'fage', 'fail', 'fan', 'fans', 'farm', 'fast', 'fi', 'fiat',
    'fido', 'film', 'fire', 'fish', 'fit', 'fj', 'fk', 'flir', 'fly', 'fm',
    'fo', 'foo', 'food', 'ford', 'fox', 'fr', 'free', 'frl', 'ftr', 'fun',
    'fund', 'fyi', 'ga', 'gal', 'game', 'gap', 'gb', 'gbiz', 'gd', 'gdn', 'ge',
    'gea', 'gent', 'gf', 'gg', 'ggee', 'gh', 'gi', 'gift', 'gl', 'gle', 'gm',
    'gmbh', 'gmo', 'gmx', 'gn', 'gold', 'golf', 'goo', 'goog', 'gop', 'got',
    'gov', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'guge', 'guru', 'gw', 'gy',
    'hair', 'haus', 'hbo', 'hdfc', 'help', 'here', 'hgtv', 'hiv', 'hk', 'hkt',
    'hm', 'hn', 'host', 'hot', 'how', 'hr', 'hsbc', 'ht', 'hu', 'ibm', 'icbc',
    'ice', 'icu', 'id', 'ie', 'ieee', 'ifm', 'il', 'im', 'imdb', 'immo', 'in',
    'info', 'ing', 'ink', 'int', 'io', 'iq', 'ir', 'is', 'ist', 'it', 'itau',
    'itv', 'iwc', 'java', 'jcb', 'jcp', 'je', 'jeep', 'jio', 'jlc', 'jll',
    'jm', 'jmp', 'jnj', 'jo', 'jobs', 'jot', 'joy', 'jp', 'jprs', 'kddi', 'ke',
    'kfh', 'kg', 'kh', 'ki', 'kia', 'kim', 'kiwi', 'km', 'kn', 'kp', 'kpmg',
    'kpn', 'kr', 'krd', 'kred', 'kw', 'ky', 'kz', 'la', 'land', 'lat', 'law',
    'lb', 'lc', 'lds', 'lego', 'lgbt', 'li', 'lidl', 'life', 'like', 'limo',
    'link', 'live', 'lk', 'llc', 'loan', 'loft', 'lol', 'love', 'lpl', 'lr',
    'ls', 'lt', 'ltd', 'ltda', 'lu', 'luxe', 'lv', 'ly', 'ma', 'maif', 'man',
    'map', 'mba', 'mc', 'md', 'me', 'med', 'meet', 'meme', 'men', 'menu', 'mg',
    'mh', 'mil', 'mini', 'mint', 'mit', 'mk', 'ml', 'mlb', 'mls', 'mm', 'mma',
    'mn', 'mo', 'mobi', 'moda', 'moe', 'moi', 'mom', 'moto', 'mov', 'mp', 'mq',
    'mr', 'ms', 'msd', 'mt', 'mtn', 'mtr', 'mu', 'mv', 'mw', 'mx', 'my', 'mz',
    'na', 'nab', 'name', 'navy', 'nba', 'nc', 'ne', 'nec', 'net', 'new',
    'news', 'next', 'nf', 'nfl', 'ng', 'ngo', 'nhk', 'ni', 'nico', 'nike',
    'nl', 'no', 'now', 'np', 'nr', 'nra', 'nrw', 'ntt', 'nu', 'nyc', 'nz',
    'obi', 'off', 'ollo', 'om', 'one', 'ong', 'onl', 'ooo', 'open', 'org',
    'ott', 'ovh', 'pa', 'page', 'pars', 'pay', 'pccw', 'pe', 'pet', 'pf', 'pg',
    'ph', 'phd', 'pics', 'pid', 'pin', 'ping', 'pink', 'pk', 'pl', 'play',
    'plus', 'pm', 'pn', 'pnc', 'pohl', 'porn', 'post', 'pr', 'pro', 'prod',
    'prof', 'pru', 'ps', 'pt', 'pub', 'pw', 'pwc', 'py', 'qa', 'qpon', 'qvc',
    'raid', 're', 'read', 'red', 'reit', 'ren', 'rent', 'rest', 'rich', 'ril',
    'rio', 'rip', 'rmit', 'ro', 'room', 'rs', 'rsvp', 'ru', 'ruhr', 'run',
    'rw', 'rwe', 'sa', 'safe', 'sale', 'sap', 'sarl', 'sas', 'save', 'saxo',
    'sb', 'sbi', 'sbs', 'sc', 'sca', 'scb', 'scor', 'scot', 'sd', 'se', 'seat',
    'seek', 'ses', 'sew', 'sex', 'sexy', 'sfr', 'sg', 'sh', 'shaw', 'shia',
    'shop', 'show', 'si', 'silk', 'sina', 'site', 'sj', 'sk', 'ski', 'skin',
    'sky', 'sl', 'sm', 'sn', 'sncf', 'so', 'sohu', 'song', 'sony', 'soy',
    'spot', 'sr', 'srl', 'srt', 'st', 'star', 'stc', 'su', 'surf', 'sv', 'sx',
    'sy', 'sz', 'tab', 'talk', 'tax', 'taxi', 'tc', 'tci', 'td', 'tdk', 'team',
    'tech', 'tel', 'teva', 'tf', 'tg', 'th', 'thd', 'tiaa', 'tips', 'tj',
    'tjx', 'tk', 'tl', 'tm', 'tn', 'to', 'top', 'town', 'toys', 'tr', 'trv',
    'tt', 'tube', 'tui', 'tv', 'tvs', 'tw', 'tz', 'ua', 'ubs', 'ug', 'uk',
    'uno', 'uol', 'ups', 'us', 'uy', 'uz', 'va', 'vana', 'vc', 've', 'vet',
    'vg', 'vi', 'vig', 'vin', 'vip', 'visa', 'viva', 'vivo', 'vn', 'vote',
    'voto', 'vu', 'wang', 'wed', 'weir', 'wf', 'wien', 'wiki', 'win', 'wine',
    'wme', 'work', 'wow', 'ws', 'wtc', 'wtf', 'xbox', 'xin', 'xxx', 'xyz',
    'ye', 'yoga', 'you', 'yt', 'yun', 'za', 'zara', 'zero', 'zip', 'zm',
    'zone', 'zw'
]
