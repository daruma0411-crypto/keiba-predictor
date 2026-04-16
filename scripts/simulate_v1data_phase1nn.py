"""
回路遮断テスト: V1データ + Phase1 NN
V1の features_v9b_cache.pkl を Phase1の PredictorPhase1 (3層, 純NLL) で学習。
V1データが正しければ V1相当の精度が出るはず。
出なければ predictor_v2.py の前処理にバグがある。
"""
import sys, io, os, warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir('C:/Users/iwash/keiba-predictor')
sys.path.insert(0, 'C:/Users/iwash/keiba-predictor')

import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
np.random.seed(42)

# V1 NNモデル (simulate_v9b.pyから完全コピー)
class HorseRaceModel(nn.Module):
    def __init__(self, nf, ed=None):
        super().__init__()
        self.embs = nn.ModuleDict()
        te = 0
        if ed:
            for n, (v, d) in ed.items():
                self.embs[n] = nn.Embedding(v, d); te += d
        self.net = nn.Sequential(
            nn.Linear(nf+te, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, 1)
        self.sig = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
    def forward(self, x, c=None):
        p = [x]
        if c:
            for n, i in c.items():
                if n in self.embs: p.append(self.embs[n](i))
        h = self.net(torch.cat(p, 1))
        return self.mu(h), self.sig(h) + 0.1

class Predictor:
    def __init__(self, nf, cf=None):
        self.nf, self.cf = nf, cf or []
        self.sc = StandardScaler(); self.les = {}; self.m = None
        self.d = torch.device('cpu'); self._med = None
    def _p(self, df, fit=False):
        X = df[self.nf].copy()
        for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
        if fit:
            self._med = X.median(); X = X.fillna(self._med); Xs = self.sc.fit_transform(X)
        else:
            X = X.fillna(self._med); Xs = self.sc.transform(X)
        Xt = torch.FloatTensor(Xs).to(self.d)
        ct = {}
        for c in self.cf:
            if c in df.columns:
                v = df[c].fillna('unknown').astype(str)
                if fit: le = LabelEncoder(); le.fit(v); self.les[c] = le
                else:
                    le = self.les[c]; v = v.map(lambda x: x if x in le.classes_ else le.classes_[0])
                ct[c] = torch.LongTensor(le.transform(v)).to(self.d)
        return Xt, ct
    def train(self, df, ep=50, lr=0.003, bs=256, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        df = df[df['finish']>0].copy()
        y = df['finish'].values.astype(float); h = df['heads'].values.astype(float); h[h==0]=16
        yt = torch.FloatTensor((y-1)/(h-1)).unsqueeze(1).to(self.d)
        Xt, ct = self._p(df, True)
        ed = {}
        for c in self.cf:
            if c in self.les: vs = len(self.les[c].classes_); ed[c] = (vs, min(50, max(4,(vs+1)//2)))
        self.m = HorseRaceModel(len(self.nf), ed).to(self.d)
        opt = torch.optim.Adam(self.m.parameters(), lr=lr); self.m.train(); n = len(Xt)
        for e in range(ep):
            idx = torch.randperm(n)
            for i in range(0, n, bs):
                bi = idx[i:i+bs]; mu, s = self.m(Xt[bi], {c:t[bi] for c,t in ct.items()})
                l = torch.mean(0.5*torch.log(s**2)+0.5*((yt[bi]-mu)/s)**2)
                opt.zero_grad(); l.backward(); opt.step()
    def predict(self, df):
        self.m.eval(); Xt, ct = self._p(df, False)
        with torch.no_grad(): mu, s = self.m(Xt, ct)
        return pd.DataFrame({'mu':mu.squeeze().numpy(),'sigma':s.squeeze().numpy(),
            'horse_name':df['horse_name'].values,'ketto_num':df['ketto_num'].values,
            'umaban':df['umaban'].values,'odds':df['odds'].values if 'odds' in df.columns else np.nan})

def run():
    from src.qmc_courses import qmc_sim

    print('='*60)
    print('  回路遮断テスト: V1データ + V1互換NN')
    print('  simulate_v9b.py と同一ロジック（モジュール非依存版）')
    print('='*60)

    feat = pd.read_pickle('data/features_v9b_cache.pkl')
    print(f'V1 cache: {len(feat):,} rows, {feat["date"].min()} ~ {feat["date"].max()}')

    # V1キャッシュにある特徴量のみ使用
    FEATURES = [
        'wakuban','futan','bataijyu','zogen_sa','heads',
        'past_count','ema_time_zscore','ema_finish',
        'win_rate','top3_rate','avg_run_style',
        'same_dist_finish','same_surface_finish','interval_days',
        'jockey_win_rate','jockey_top3_rate',
        'trainer_win_rate','trainer_top3_rate',
        'avg_jyuni_3c','avg_jyuni_4c',
        'prev_race_class','log_prize_money','weighted_ema_finish',
        'ema_agari','long_stretch_avg','prev_dist_diff',
    ]
    CAT = ['kisyu_code','chokyosi_code','banusi_code','sire_code']

    use_f = [f for f in FEATURES if f in feat.columns]
    use_cat = [c for c in CAT if c in feat.columns]
    print(f'Features: {len(use_f)}, Cat: {use_cat}')

    # 桜花賞特定: 阪神(09) 4月 11R
    sakura_ids = {}
    apr = feat[(feat['date'].dt.month==4) & (feat['is_turf']==1)]
    for rid in apr['race_id'].unique():
        parts = rid.split('_')
        if len(parts)==5 and parts[1]=='09' and parts[4]=='11':
            g = apr[apr['race_id']==rid]
            if len(g) >= 16:
                sakura_ids[int(parts[0][:4])] = rid

    print(f'Sakura: {sorted(sakura_ids.keys())}')

    results = []
    for year in range(2016, 2026):
        if year not in sakura_ids: continue
        rid = sakura_ids[year]
        rf = feat[feat['race_id']==rid].copy()
        if len(rf)==0: continue
        rdate = rf['date'].iloc[0]

        tr = feat[(feat['date']<rdate)&(feat['is_turf']==1)&(feat['past_count']>0)&(feat['finish']>0)].sort_values('date').tail(20000).copy()

        print(f'\n{year}年桜花賞 ({rdate.strftime("%Y-%m-%d")}):')
        print(f'  学習: {len(tr):,} | 出走: {len(rf)} heads')

        pred = Predictor(use_f, use_cat)
        pred.train(tr, ep=50, lr=0.003, seed=42)
        ps = pred.predict(rf)

        mu_range = ps['mu'].max()-ps['mu'].min()
        mu_std = ps['mu'].std()
        print(f'  μ range={mu_range:.4f}, std={mu_std:.4f}')

        mc = qmc_sim(ps, race_features=rf, course='hanshin_turf_1600_outer', n=100000)

        act = rf.sort_values('finish').head(3)
        a3 = set(act['umaban'].astype(int)); a1 = int(act.iloc[0]['umaban'])
        t5 = mc.head(5); t5u = set(t5['umaban'].astype(int))
        ov = len(t5u & a3); w = int(t5.iloc[0]['umaban'])==a1

        t5f = []
        print(f'  予測TOP5:')
        for rk,(_, r) in enumerate(t5.iterrows(),1):
            u = int(r['umaban']); a = rf[rf['umaban']==u]
            af = int(a.iloc[0]['finish']) if len(a)>0 else 18; t5f.append(af)
            h = '*' if u in a3 else ' '
            mu_val = ps[ps['umaban']==u]['mu'].values[0] if len(ps[ps['umaban']==u])>0 else 0
            print(f'    {h}{rk}位 [{u:2d}] {r["horse_name"]:16s} 勝率{r["win_prob"]:5.1%} 複勝{r["top3_prob"]:5.1%} μ={mu_val:.4f} (実際{af}着)')
        print(f'  実際TOP3:')
        for _, r in act.iterrows():
            print(f'    {int(r["finish"])}着 [{int(r["umaban"]):2d}] {r["horse_name"]}')

        avg = np.mean(t5f)
        ana = any(r['odds']>=10 and int(r['umaban']) in a3 for _,r in t5.iterrows() if pd.notna(r.get('odds')))
        results.append({'year':year,'win':w,'overlap':ov,'t5':a1 in t5u,'avg':avg,'ana':ana,'p1f':t5f[0]})
        sys.stdout.flush()

    n = len(results)
    print(f'\n{"="*60}')
    print(f'  回路遮断テスト KPIレポート')
    print(f'{"="*60}')
    wn = sum(r['win'] for r in results); wr = wn/n*100
    t5ok = sum(1 for r in results if r['avg']<=9)
    an = sum(r['ana'] for r in results)
    o2 = sum(1 for r in results if r['overlap']>=2); ovr = o2/n*100

    print(f'\n  ■ KPI 1: 1着的中率: {wn}/{n} ({wr:.0f}%)')
    print(f'  ■ KPI 2: TOP5平均9着以内: {t5ok}/{n}')
    print(f'  ■ KPI 3: 穴馬検知: {an}/{n}年')
    print(f'  ■ KPI 4: 馬券内占有率: {o2}/{n} ({ovr:.0f}%)')

    print(f'\n  年別:')
    for r in results:
        m = '◎' if r['win'] else ('○' if r['t5'] else '×')
        a = '穴★' if r['ana'] else '    '
        print(f'    {r["year"]}: {m} 1位→{r["p1f"]}着 3着内{r["overlap"]}/3 TOP5平均{r["avg"]:.1f}着 {a}')

    print(f'\n  === 3モデル比較 ===')
    print(f'  V1(v9b本番):           的中~30%, 占有率80%')
    print(f'  V2(ListNet+4層+CSV):    的中 0%, 占有率10%')
    print(f'  Phase1(NLL+3層+CSV):    的中 0%, 占有率10%')
    print(f'  回路遮断(NLL+3層+V1データ): 的中{wn}/{n}({wr:.0f}%), 占有率{ovr:.0f}%')

if __name__ == '__main__':
    run()
