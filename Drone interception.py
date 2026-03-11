# ── stdlib ────────────────────────────────────────────────────────────────────
import math, time, warnings
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional, List, Dict, Tuple

# ── numerics ──────────────────────────────────────────────────────────────────
import numpy as np
from numpy.linalg import norm, inv, slogdet
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, lfilter

# ── plotting ──────────────────────────────────────────────────────────────────
import matplotlib
_BACKENDS = ["TkAgg","Qt5Agg","Qt6Agg","WxAgg","MacOSX","Agg"]
for _b in _BACKENDS:
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as _chk; _chk.figure(); _chk.close("all")
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Wedge, Circle, Rectangle
from matplotlib.patheffects import withStroke
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
plt.rcParams["toolbar"] = "None"

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS & SIMULATION CONFIG
# ══════════════════════════════════════════════════════════════════════════════
RNG = np.random.default_rng(2025)

ARENA_M          = 400
DT               = 0.08
NUM_INTERCEPTORS = 6
MAX_THREATS      = 8
WAVE_PERIOD      = 130
MAX_FRAMES       = 1200
ANIM_INTERVAL    = 45

G            = 9.81
RHO_0        = 1.225
SCALE_HEIGHT = 8500

RADAR_RANGE  = 300.0          # slightly extended
RADAR_NOISE  = 1.2
IRST_RANGE   = 200.0          # extended
IRST_NOISE   = 2.4
LIDAR_RANGE  = 140.0          # extended
LIDAR_NOISE  = 0.4

# ── KEY ACCURACY PARAMETERS ────────────────────────────────────────────────
INTERCEPT_R  = 18.0           # was 7.5 — direct-kill radius
FRAG_RADIUS  = 40.0           # was 22 — fragmentation kill radius
FRAG_KILL_PROB = 0.85         # was 0.35 — probability of kill in frag zone

# ── Colour palette ─────────────────────────────────────────────────────────
PAL = dict(
    bg       = "#030d06",
    panel    = "#040f08",
    panel2   = "#061410",
    border   = "#0e3016",
    border2  = "#1a5025",
    grid     = "#071c0a",
    green    = "#00ff55",
    green2   = "#00dd44",
    green3   = "#006622",
    green4   = "#003811",
    green5   = "#001f09",
    amber    = "#ffb800",
    amber2   = "#cc8800",
    amber3   = "#7a5200",
    red      = "#ff2244",
    red2     = "#991122",
    red3     = "#4a0810",
    cyan     = "#00ffea",
    cyan2    = "#00bbaa",
    blue     = "#1166ff",
    blue2    = "#0033aa",
    white    = "#c8ffda",
    dim      = "#2a5530",
    dim2     = "#1a3320",
    magenta  = "#ff00cc",
    orange   = "#ff6600",
    yellow   = "#ffee00",
    purple   = "#aa44ff",
)

THREAT_PALETTE  = ["#ff3333","#ff7700","#ff33bb","#ff0055",
                   "#cc33ff","#ff5500","#ffaa00","#ff0000"]
UAV_PALETTE     = ["#00ff55","#00ffea","#aaff22","#ffee00",
                   "#00aaff","#ff88ff"]

# ══════════════════════════════════════════════════════════════════════════════
#  TERRAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class Terrain:
    GRID = 200

    def __init__(self):
        G = self.GRID
        self.cell = ARENA_M / G
        elev = np.zeros((G, G))
        rng  = np.random.default_rng(7)
        for oct_, amp, freq in [(1,35,0.015),(2,18,0.04),(3,8,0.09),(4,3,0.20)]:
            noise = rng.uniform(-1,1,(G,G))
            noise = gaussian_filter(noise, sigma=G/(freq*G*2))
            elev += amp * noise
        elev -= elev.min(); elev /= elev.max(); elev = elev*80+5

        cx = np.linspace(0,G,100); cy = G//2+20*np.sin(cx*0.06)
        for i in range(len(cx)-1):
            x0,y0 = int(cx[i]),int(cy[i])
            for dx in range(-8,9):
                for dy in range(-8,9):
                    xi,yi = x0+dx,y0+dy
                    if 0<=xi<G and 0<=yi<G:
                        d = math.sqrt(dx**2+dy**2)
                        elev[yi,xi] -= max(0,(10-d)*1.5)

        uc = G//2
        for i in range(G):
            for j in range(G):
                d = math.sqrt((i-uc)**2+(j-uc)**2)
                if d < G*0.12: elev[i,j] = max(elev[i,j],15+5*rng.uniform())

        self.elev   = np.clip(elev,0,90)
        self.grad_x = np.gradient(self.elev, axis=1)
        self.grad_y = np.gradient(self.elev, axis=0)

        feat = np.zeros((G,G),dtype=int)
        for i in range(G):
            for j in range(G):
                d = math.sqrt((i-uc)**2+(j-uc)**2)
                if d < G*0.12: feat[i,j]=1
                elif elev[i,j]>55 and rng.random()<0.6: feat[i,j]=2
                elif elev[i,j]<8: feat[i,j]=3
        self.feat = feat
        self._build_image()
        self.heatmap = np.zeros((G,G), dtype=float)

    def _build_image(self):
        G = self.GRID
        img = np.zeros((G,G,4))
        feat_colors = {0:(0.05,0.12,0.07),1:(0.15,0.18,0.20),
                       2:(0.03,0.16,0.05),3:(0.02,0.06,0.20)}
        for i in range(G):
            for j in range(G):
                fc = feat_colors[self.feat[i,j]]
                shade = 0.35+0.65*(self.elev[i,j]/90)
                img[i,j,:3] = [c*shade for c in fc]
                img[i,j,3]  = 0.85
        az,al = np.radians(315),np.radians(45)
        hs = (np.cos(al)*np.cos(np.arctan(np.sqrt(self.grad_x**2+self.grad_y**2)))
              +np.sin(al)*(self.grad_x*np.cos(az)+self.grad_y*np.sin(az)))
        hs = np.clip((hs+1)*0.5,0.2,0.95)
        for c in range(3): img[:,:,c] *= hs
        self.image = img

    def height_at(self,x,y):
        gx = np.clip(x/self.cell,0,self.GRID-1.001)
        gy = np.clip(y/self.cell,0,self.GRID-1.001)
        xi,yi = int(gx),int(gy); fx,fy=gx-xi,gy-yi
        h00=self.elev[yi,xi]
        h10=self.elev[yi,min(xi+1,self.GRID-1)]
        h01=self.elev[min(yi+1,self.GRID-1),xi]
        h11=self.elev[min(yi+1,self.GRID-1),min(xi+1,self.GRID-1)]
        return h00*(1-fx)*(1-fy)+h10*fx*(1-fy)+h01*(1-fx)*fy+h11*fx*fy

    def los_clear(self,p1,h1,p2,h2,steps=20):
        for t in np.linspace(0,1,steps):
            px=p1[0]+t*(p2[0]-p1[0]); py=p1[1]+t*(p2[1]-p1[1])
            ph=h1+t*(h2-h1)
            if ph < self.height_at(px,py)-1.0: return False
        return True

    def radar_clutter(self,pos):
        feat=self.feat[int(np.clip(pos[1]/self.cell,0,self.GRID-1)),
                       int(np.clip(pos[0]/self.cell,0,self.GRID-1))]
        return {0:0.1,1:0.55,2:0.35,3:0.05}[feat]

    def accumulate_heat(self, pos, value=1.0):
        gx = int(np.clip(pos[0]/self.cell, 0, self.GRID-1))
        gy = int(np.clip(pos[1]/self.cell, 0, self.GRID-1))
        self.heatmap[gy, gx] += value
        if gx > 0: self.heatmap[gy, gx-1] += value*0.4
        if gx < self.GRID-1: self.heatmap[gy, gx+1] += value*0.4
        if gy > 0: self.heatmap[gy-1, gx] += value*0.4
        if gy < self.GRID-1: self.heatmap[gy+1, gx] += value*0.4

TERRAIN = Terrain()

# ══════════════════════════════════════════════════════════════════════════════
#  ATMOSPHERE & WIND
# ══════════════════════════════════════════════════════════════════════════════
class Atmosphere:
    def __init__(self):
        self.mean_wind = np.array([5.6,-5.6])
        self.phase  = RNG.uniform(0,2*np.pi,(4,4))
        self.amp    = RNG.uniform(1.5,4.0,(4,4))
        self.fq     = RNG.uniform(0.005,0.02,(4,4))
        self.t      = 0.0
        self._turb  = np.zeros(2)
        b,a = butter(1,0.04)
        self._b,self._a = b,a
        self._zf = np.zeros((max(len(a),len(b))-1,2))

    def step(self):
        self.t += DT
        raw = RNG.normal(0,2.5,2)
        self._turb,self._zf = lfilter(self._b,self._a,
                                       raw.reshape(1,2),zi=self._zf,axis=0)
        self._turb = self._turb[0]

    def wind_at(self,pos,alt):
        wx = sum(self.amp[i,j]*np.sin(self.fq[i,j]*pos[0]+self.phase[i,j]+self.t*0.3)
                 for i in range(4) for j in range(4))/8
        wy = sum(self.amp[i,j]*np.cos(self.fq[i,j]*pos[1]+self.phase[i,j]+self.t*0.25)
                 for i in range(4) for j in range(4))/8
        shear = 1.0+0.3*(alt/50.0)
        return (self.mean_wind+np.array([wx,wy])+self._turb)*shear

    def density(self,alt):
        return RHO_0*math.exp(-alt/SCALE_HEIGHT)

ATM = Atmosphere()

# ══════════════════════════════════════════════════════════════════════════════
#  SENSOR FUSION
# ══════════════════════════════════════════════════════════════════════════════
class SensorType(IntEnum):
    RADAR=0; IRST=1; LIDAR=2; EO_IR=3; RWR=4

@dataclass
class SensorReturn:
    sensor:    SensorType
    pos:       np.ndarray
    vel:       Optional[np.ndarray]
    quality:   float
    threat_id: int
    timestamp: float

class SensorFusion:
    def __init__(self):
        self.radar_jammed = False
        self.jam_duration = 0
        self.sig_radar = 0.0
        self.sig_irst  = 0.0
        self.sig_eo    = 0.0

    def _radar_pd(self,rcs,dist,alt,clutter,jamming):
        # FIX: ECM/jamming now only reduces Pd by 50%, not down to near-zero
        if jamming:
            snr=(rcs*1e4)/(dist**3.5+1e-6)*ATM.density(alt)/RHO_0/(1+clutter*3)
            return max(0.0, (1.0-math.exp(-snr*12)) * 0.50)   # 50% degradation
        snr=(rcs*1e4)/(dist**3.5+1e-6)*ATM.density(alt)/RHO_0/(1+clutter*3)
        return 1.0-math.exp(-snr*12)

    def observe(self,threat,obs_pos,obs_alt,time_now):
        returns=[]; dist=norm(threat.pos-obs_pos); alt=threat.altitude
        clutter=TERRAIN.radar_clutter(threat.pos)
        los=TERRAIN.los_clear(obs_pos,obs_alt,threat.pos,alt)

        if dist<RADAR_RANGE and los:
            pd=self._radar_pd(threat.rcs,dist,alt,clutter,self.radar_jammed)
            self.sig_radar = pd*(1-clutter*0.5)
            if RNG.random()<pd:
                noise=RADAR_NOISE*(1+clutter*2)/(threat.rcs+0.1)
                meas=threat.pos+RNG.normal(0,noise,2)
                vel_m=(threat.vel+RNG.normal(0,1.8,2)) if hasattr(threat,'vel') else None
                q=pd*(1-clutter*0.5)
                returns.append(SensorReturn(SensorType.RADAR,meas,vel_m,q,threat.uid,time_now))
        else:
            self.sig_radar = max(0, self.sig_radar-0.05)

        if dist<IRST_RANGE and los:
            heat=threat.heat_sig; pd_irst=1-math.exp(-heat*dist/(IRST_RANGE*2))
            self.sig_irst = pd_irst*0.7
            if RNG.random()<pd_irst:
                noise=IRST_NOISE*(dist/IRST_RANGE)
                meas=threat.pos+RNG.normal(0,noise,2)
                returns.append(SensorReturn(SensorType.IRST,meas,None,pd_irst*0.7,threat.uid,time_now))
        else:
            self.sig_irst = max(0, self.sig_irst-0.05)

        if dist<LIDAR_RANGE and los:
            meas=threat.pos+RNG.normal(0,LIDAR_NOISE,2)
            vel_m=threat.vel+RNG.normal(0,0.6,2)
            self.sig_eo = 0.95
            returns.append(SensorReturn(SensorType.EO_IR,meas,vel_m,0.95,threat.uid,time_now))
        else:
            self.sig_eo = max(0, self.sig_eo-0.05)

        return returns

    def fuse(self,returns):
        if not returns: return None,None
        weights=np.array([r.quality for r in returns]); weights/=weights.sum()
        pos_fused=sum(w*r.pos for w,r in zip(weights,returns))
        noise_map={SensorType.RADAR:RADAR_NOISE,SensorType.IRST:IRST_NOISE,
                   SensorType.LIDAR:LIDAR_NOISE,SensorType.EO_IR:LIDAR_NOISE*0.7,
                   SensorType.RWR:RADAR_NOISE*3}
        cov=np.zeros((2,2))
        for w,r in zip(weights,returns):
            sig=noise_map.get(r.sensor,2.0); cov+=w*np.eye(2)*sig**2
        return pos_fused,cov

FUSION=SensorFusion()

# ══════════════════════════════════════════════════════════════════════════════
#  IMM-EKF TRACKER
# ══════════════════════════════════════════════════════════════════════════════
def _make_F_cv(dt):
    F=np.eye(4); F[0,2]=dt; F[1,3]=dt; return F

def _make_F_ca(dt):
    F=np.eye(6); dt2=0.5*dt**2
    F[0,2]=dt; F[0,4]=dt2; F[1,3]=dt; F[1,5]=dt2
    F[2,4]=dt; F[3,5]=dt; return F

def _make_F_ct(dt,omega=0.15):
    if abs(omega)<1e-4: return _make_F_cv(dt)[:4,:4]
    s,c=math.sin(omega*dt),math.cos(omega*dt); F=np.eye(5)
    F[0,2]=s/omega; F[0,3]=-(1-c)/omega; F[1,2]=(1-c)/omega; F[1,3]=s/omega
    F[2,2]=c; F[2,3]=-s; F[3,2]=s; F[3,3]=c; return F

def _make_F_singer(dt,alpha=0.5):
    e=math.exp(-alpha*dt); F=np.eye(6)
    F[0,2]=dt; F[0,4]=(e+alpha*dt-1)/alpha**2
    F[1,3]=dt; F[1,5]=(e+alpha*dt-1)/alpha**2
    F[2,4]=(1-e)/alpha; F[3,5]=(1-e)/alpha; F[4,4]=e; F[5,5]=e; return F

class KalmanModel:
    def __init__(self,name,n,F_fn,Q_base,H,R):
        self.name=name; self.n=n; self.F_fn=F_fn; self.Q_b=Q_base
        self.H=H; self.R=R; self.x=np.zeros(n); self.P=np.eye(n)*50

    def init(self,pos,vel=None):
        self.x[:2]=pos
        if vel is not None and self.n>=4: self.x[2:4]=vel
        self.P=np.eye(self.n)*50

    def predict(self,dt):
        F=self.F_fn(dt)
        if F.shape[0]!=self.n:
            F2=np.eye(self.n); s=min(F.shape[0],self.n); F2[:s,:s]=F[:s,:s]; F=F2
        Q=np.eye(self.n)*self.Q_b
        self.x=F@self.x; self.P=F@self.P@F.T+Q

    def update(self,z,R_override=None):
        R=R_override if R_override is not None else self.R
        H=self.H
        if H.shape[1]!=self.n:
            H2=np.zeros((2,self.n)); H2[:,:min(2,self.n)]=H[:,:min(2,self.n)]; H=H2
        inn=z-H@self.x; S=H@self.P@H.T+R
        try: Si=inv(S)
        except: Si=np.eye(2)*0.01
        K=self.P@H.T@Si; self.x=self.x+K@inn
        self.P=(np.eye(self.n)-K@H)@self.P
        sign,logdet=slogdet(S); maha=float(inn@Si@inn)
        lik=math.exp(-0.5*(maha+logdet+2*math.log(2*math.pi)))
        return max(lik,1e-300)

    def pos(self): return self.x[:2].copy()
    def vel(self): return self.x[2:4].copy() if self.n>=4 else np.zeros(2)
    def predict_pos(self,steps,dt):
        x=self.x.copy(); F=self.F_fn(dt)
        if F.shape[0]!=self.n:
            F2=np.eye(self.n); s=min(F.shape[0],self.n); F2[:s,:s]=F[:s,:s]; F=F2
        for _ in range(steps): x=F@x
        return x[:2]

H2=np.zeros((2,6)); H2[0,0]=1; H2[1,1]=1
H4=np.zeros((2,4)); H4[0,0]=1; H4[1,1]=1
R_std=np.eye(2)*RADAR_NOISE**2

class IMMTracker:
    TRANS=np.array([[0.75,0.10,0.10,0.05],[0.10,0.70,0.12,0.08],
                    [0.10,0.12,0.70,0.08],[0.05,0.10,0.10,0.75]])

    def __init__(self,pos,vel):
        self._omega=0.15
        self.models=[
            KalmanModel("CV",4,_make_F_cv,0.25,H4.copy(),R_std),
            KalmanModel("CA",6,_make_F_ca,1.20,H2.copy(),R_std),
            KalmanModel("CT",5,lambda dt:_make_F_ct(dt,self._omega),2.50,np.zeros((2,5)),R_std),
            KalmanModel("SG",6,lambda dt:_make_F_singer(dt,0.4),3.00,H2.copy(),R_std),
        ]
        self.models[2].H=np.zeros((2,5)); self.models[2].H[0,0]=1; self.models[2].H[1,1]=1
        self.mu=np.array([0.40,0.25,0.20,0.15])
        for m in self.models: m.init(pos,vel)
        self.est_hist=deque(maxlen=180); self.meas_hist=deque(maxlen=180)
        self.true_hist=deque(maxlen=180); self.mode_hist=deque(maxlen=180)
        self.innovation=deque(maxlen=50)

    def step(self,z,R_meas=None,true_pos=None):
        N=len(self.models)
        mu_bar=self.TRANS.T@self.mu; mu_bar=np.clip(mu_bar,1e-8,1.0); mu_bar/=mu_bar.sum()
        M=(self.TRANS*self.mu[:,None])/(mu_bar[None,:]+1e-12)
        x_mix=[sum(M[i,j]*self.models[i].x[:self.models[j].n]
                   if self.models[i].n>=self.models[j].n
                   else np.pad(M[i,j]*self.models[i].x,(0,self.models[j].n-self.models[i].n))
                   for i in range(N)) for j in range(N)]
        P_mix=[]
        for j in range(N):
            nj=self.models[j].n; Pj=np.zeros((nj,nj))
            for i in range(N):
                ni=self.models[i].n; xi=self.models[i].x; xj=x_mix[j]
                dx=xi[:nj]-xj if ni>=nj else np.pad(xi,(0,nj-ni))-xj
                Pi=self.models[i].P; Pi_ext=np.zeros((nj,nj)); s=min(ni,nj); Pi_ext[:s,:s]=Pi[:s,:s]
                Pj+=M[i,j]*(Pi_ext+np.outer(dx,dx))
            P_mix.append(Pj)
        likelihoods=[]
        for j,m in enumerate(self.models):
            m.x[:]=x_mix[j]; m.P[:]=P_mix[j]; m.predict(DT); lik=m.update(z,R_meas); likelihoods.append(lik)
        self._omega=float(np.clip(self.models[2].x[4],-0.6,0.6)) if self.models[2].n>4 else 0.15
        c=np.array(likelihoods)*mu_bar; cs=c.sum()
        self.mu=c/(cs if cs>0 else 1.0); self.mu=np.clip(self.mu,1e-4,1.0); self.mu/=self.mu.sum()
        inn=z-self.fpos(); self.innovation.append(float(norm(inn)))
        self.est_hist.append(self.fpos().copy()); self.meas_hist.append(z.copy())
        self.mode_hist.append(self.dominant_model())
        if true_pos is not None: self.true_hist.append(true_pos.copy())

    def fpos(self): return sum(self.mu[i]*self.models[i].pos() for i in range(4))
    def fvel(self): return sum(self.mu[i]*self.models[i].vel() for i in range(4))
    def future(self,steps):
        return sum(self.mu[i]*self.models[i].predict_pos(steps,DT) for i in range(4))
    def dominant_model(self): return self.models[int(np.argmax(self.mu))].name
    def rmse(self):
        if len(self.true_hist)<2 or len(self.est_hist)<2: return 0.0
        errs=[norm(e-t) for e,t in zip(list(self.est_hist),list(self.true_hist))]
        return float(np.sqrt(np.mean(np.array(errs)**2)))

# ══════════════════════════════════════════════════════════════════════════════
#  THREAT ARCHETYPES
# ══════════════════════════════════════════════════════════════════════════════
class ThreatType(IntEnum):
    QUADCOPTER=0; FIXED_WING=1; SWARM_NODE=2; KAMIKAZE=3
    STEALTH_UAV=4; LOITERING=5; HYPERSONIC=6

THREAT_SPECS={
    ThreatType.QUADCOPTER: dict(spd=(7,11),   accel=8,  rcs=0.8, heat=0.6, mass=1.5, drag=0.35),
    ThreatType.FIXED_WING: dict(spd=(14,20),  accel=12, rcs=1.2, heat=0.8, mass=3.0, drag=0.12),
    ThreatType.SWARM_NODE: dict(spd=(9,13),   accel=10, rcs=0.4, heat=0.4, mass=0.8, drag=0.28),
    ThreatType.KAMIKAZE:   dict(spd=(18,24),  accel=15, rcs=0.9, heat=1.2, mass=4.0, drag=0.18),
    ThreatType.STEALTH_UAV:dict(spd=(12,18),  accel=9,  rcs=0.05,heat=0.2, mass=2.5, drag=0.14),
    ThreatType.LOITERING:  dict(spd=(10,14),  accel=6,  rcs=0.6, heat=0.7, mass=5.0, drag=0.20),
    ThreatType.HYPERSONIC: dict(spd=(35,50),  accel=4,  rcs=0.3, heat=2.0, mass=8.0, drag=0.06),
}

_global_tid=0

class Threat:
    def __init__(self,ttype,wave=1,swarm_leader=None):
        global _global_tid
        _global_tid+=1
        self.uid=_global_tid; self.ttype=ttype
        self.color=THREAT_PALETTE[int(ttype)%len(THREAT_PALETTE)]
        self.wave=wave; self.active=True; self.intercepted=False; self.age=0

        spec=THREAT_SPECS[ttype]
        self.max_speed=float(RNG.uniform(*spec["spd"])); self.max_accel=spec["accel"]
        self.rcs=spec["rcs"]*RNG.uniform(0.8,1.2); self.heat_sig=spec["heat"]
        self.mass=spec["mass"]; self.drag_coef=spec["drag"]
        self.has_ecm=(ttype==ThreatType.STEALTH_UAV or RNG.random()<0.2)
        self.ecm_active=False

        margin=30; edge=RNG.integers(4)
        if   edge==0: p=[RNG.uniform(margin,ARENA_M-margin),margin]
        elif edge==1: p=[RNG.uniform(margin,ARENA_M-margin),ARENA_M-margin]
        elif edge==2: p=[margin,RNG.uniform(margin,ARENA_M-margin)]
        else:         p=[ARENA_M-margin,RNG.uniform(margin,ARENA_M-margin)]
        self.pos=np.array(p,dtype=float)
        self.altitude=TERRAIN.height_at(*self.pos)+RNG.uniform(8,35)
        self.alt_rate=0.0; self.target_alt=20.0+RNG.uniform(-5,10)

        hq=np.array([ARENA_M/2,ARENA_M/2]); direction=(hq-self.pos)
        direction/=norm(direction)+1e-8
        angle_off=RNG.uniform(-0.25,0.25); c,s=math.cos(angle_off),math.sin(angle_off)
        direction=np.array([c*direction[0]-s*direction[1],s*direction[0]+c*direction[1]])
        self.vel=direction*self.max_speed

        self.manoeuvre_timer=RNG.integers(15,60); self.manoeuvre_phase=0.0; self.manoeuvre_type=0
        self.evasion_active=False; self.evasion_target=None
        self.loiter_angle=0.0; self.dive_started=False; self.swarm_leader=swarm_leader

        self.trail_2d=deque(maxlen=90); self.trail_alt=deque(maxlen=90)
        self._priority_cache=0.0
        self.threat_score = 0.0
        self.danger_level = 0

    def step(self,interceptors,all_threats):
        if not self.active: return
        self.age+=1
        self.trail_2d.append(self.pos.copy()); self.trail_alt.append(self.altitude)
        TERRAIN.accumulate_heat(self.pos, 0.1)

        wind=ATM.wind_at(self.pos,self.altitude); rho=ATM.density(self.altitude)
        self.ecm_active=self.has_ecm and self._threat_detected(interceptors)

        hq=np.array([ARENA_M/2,ARENA_M/2]); to_hq=hq-self.pos; dist_hq=norm(to_hq)

        if self.ttype==ThreatType.LOITERING: self._loitering_guidance(dist_hq,to_hq)
        elif self.ttype==ThreatType.HYPERSONIC: self._hypersonic_guidance(to_hq)
        else: self._standard_guidance(to_hq,interceptors,all_threats)

        spd=norm(self.vel)
        if spd>0.1:
            q_dyn=0.5*rho*spd**2; drag_force=self.drag_coef*q_dyn/self.mass
            drag_accel=-(self.vel/spd)*drag_force
        else: drag_accel=np.zeros(2)
        wind_effect=(wind-self.vel)*0.04
        self.vel+=(drag_accel+wind_effect)*DT

        spd2=norm(self.vel)
        if spd2>self.max_speed*1.05: self.vel=self.vel/spd2*self.max_speed
        elif spd2<self.max_speed*0.3 and dist_hq>15: self.vel=self.vel/(spd2+1e-6)*self.max_speed*0.5
        self.pos+=self.vel*DT

        ground=TERRAIN.height_at(*self.pos); clearance=8.0
        if self.ttype in[ThreatType.KAMIKAZE,ThreatType.HYPERSONIC]:
            self.target_alt=max(ground+clearance,10+30*(dist_hq/ARENA_M))
        else: self.target_alt=max(ground+clearance,15+25*(dist_hq/ARENA_M))

        alt_error=self.target_alt-self.altitude
        self.alt_rate=np.clip(self.alt_rate+alt_error*0.15*DT,-8.0,8.0)
        self.altitude=max(ground+2,self.altitude+self.alt_rate*DT)
        self._priority_cache=self._compute_priority()

        dist_hq_now = norm(self.pos - np.array([ARENA_M/2, ARENA_M/2]))
        if dist_hq_now < 60: self.danger_level = 3
        elif dist_hq_now < 120: self.danger_level = 2
        elif dist_hq_now < 200: self.danger_level = 1
        else: self.danger_level = 0

    def _standard_guidance(self,to_hq,interceptors,all_threats):
        dist=norm(to_hq); direction=to_hq/(dist+1e-6)
        self.manoeuvre_timer-=1
        if self.manoeuvre_timer<=0:
            self.manoeuvre_type=RNG.integers(0,4); self.manoeuvre_timer=RNG.integers(20,80); self.manoeuvre_phase=0.0
        self.manoeuvre_phase+=DT; lateral=np.array([-direction[1],direction[0]])
        if self.manoeuvre_type==1: lat_cmd=lateral*math.sin(self.manoeuvre_phase*1.2)*4.5
        elif self.manoeuvre_type==2: lat_cmd=lateral*math.cos(self.manoeuvre_phase*2.0)*6.0
        elif self.manoeuvre_type==3: lat_cmd=lateral*0.3; self.target_alt+=40*DT
        else: lat_cmd=np.zeros(2)

        nearest_int=None; nearest_d=1e9
        for ic in interceptors:
            if ic.status==1 and ic.target and ic.target.uid==self.uid:
                d=norm(ic.pos-self.pos)
                if d<nearest_d: nearest_d,nearest_int=d,ic
        if nearest_int and nearest_d<55:
            away=(self.pos-nearest_int.pos)/(norm(self.pos-nearest_int.pos)+1e-6)
            evade_str=max(0,(55-nearest_d)/55)*self.max_accel
            lat_cmd+=away*evade_str; self.evasion_active=True
        else: self.evasion_active=False

        if self.ttype==ThreatType.SWARM_NODE and self.swarm_leader:
            lead_vec=self.swarm_leader.pos-self.pos; ld=norm(lead_vec)
            if ld>30: lat_cmd+=lead_vec/ld*3.0
            elif ld<10: lat_cmd-=lead_vec/ld*2.0

        desired_vel=direction*self.max_speed+lat_cmd; accel=(desired_vel-self.vel)*3.0
        am=norm(accel)
        if am>self.max_accel: accel=accel/am*self.max_accel
        self.vel+=accel*DT

    def _loitering_guidance(self,dist_hq,to_hq):
        hq=np.array([ARENA_M/2,ARENA_M/2])
        if dist_hq>60 and not self.dive_started:
            direction=to_hq/(dist_hq+1e-6); desired=direction*self.max_speed
            accel=(desired-self.vel)*2.5; am=norm(accel)
            if am>self.max_accel: accel=accel/am*self.max_accel
            self.vel+=accel*DT
        else:
            self.dive_started=True; self.loiter_angle+=DT*self.max_speed/45
            r=45; orb=hq+np.array([r*math.cos(self.loiter_angle),r*math.sin(self.loiter_angle)])
            if RNG.random()<0.003: self.target_alt=5.0
            to_orb=orb-self.pos; direction=to_orb/(norm(to_orb)+1e-6); self.vel=direction*self.max_speed

    def _hypersonic_guidance(self,to_hq):
        direction=to_hq/(norm(to_hq)+1e-6); desired=direction*self.max_speed
        accel=(desired-self.vel)*1.5; am=norm(accel)
        if am>self.max_accel: accel=accel/am*self.max_accel
        self.vel+=accel*DT

    def _threat_detected(self,interceptors):
        for ic in interceptors:
            if norm(ic.pos-self.pos)<RADAR_RANGE*0.6: return True
        return False

    def measure(self):
        noise=RADAR_NOISE*(2.0 if self.ecm_active else 1.0)/self.rcs**0.5
        return self.pos+RNG.normal(0,noise,2)

    def _compute_priority(self):
        dist_hq=norm(self.pos-np.array([ARENA_M/2,ARENA_M/2]))
        speed_factor=self.max_speed/25.0; dist_factor=1.0/(dist_hq/100+0.5)
        type_bonus={ThreatType.KAMIKAZE:3.0,ThreatType.HYPERSONIC:4.0,ThreatType.LOITERING:1.5}.get(self.ttype,1.0)
        return speed_factor*dist_factor*type_bonus*(2.0 if self.ecm_active else 1.0)

    def out_of_bounds(self):
        m=50; return(self.pos[0]<-m or self.pos[0]>ARENA_M+m
                      or self.pos[1]<-m or self.pos[1]>ARENA_M+m)

    def name_str(self):
        return ["QUAD","FWNG","SWRM","KMKZ","STLT","LOIT","HYPR"][int(self.ttype)]

# ══════════════════════════════════════════════════════════════════════════════
#  INTERCEPTOR  — KEY ACCURACY UPGRADES
# ══════════════════════════════════════════════════════════════════════════════
class Interceptor:
    def __init__(self,idx):
        angle=2*math.pi*idx/NUM_INTERCEPTORS
        # FIX: tighter patrol radius → faster scramble to threats
        r=28
        self.home=np.array([ARENA_M/2+r*math.cos(angle),ARENA_M/2+r*math.sin(angle)])
        self.pos=self.home.copy(); self.altitude=TERRAIN.height_at(*self.pos)+30.0
        self.vel=np.zeros(2); self.alt_rate=0.0
        self.color=UAV_PALETTE[idx%len(UAV_PALETTE)]; self.idx=idx
        self.status=0  # 0=idle,1=pursuing,2=returning,3=reloading,4=patrolling,5=damaged
        self.target=None; self.prev_los=None
        self.warheads=6           # FIX: was 3 → 6
        self.reload_t=0; self.kills=0; self.misses=0
        self.fuel=1.0; self.fuel_rate=0.0004   # FIX: slower fuel burn
        self.low_fuel=False
        self.seeker_lock=False; self.seeker_gimbal=0.0
        self.max_gimbal=70.0      # FIX: was 50° → 70° wider seeker cone
        # FIX: faster interceptors
        self.max_speed=42.0       # was 32
        self.max_accel=35.0       # was 22
        self.speed=0.0
        self.trail=deque(maxlen=110); self.alt_hist=deque(maxlen=110)
        self.flash=0; self.flash_r=0.0
        self.patrol_angle=angle; self.patrol_r=28.0
        self.current_streak=0
        self.total_engagements=0
        self.engage_ranges=deque(maxlen=20)

    def step(self,tracker=None):
        self.trail.append(self.pos.copy()); self.alt_hist.append(self.altitude)
        self.flash=max(0,self.flash-1); self.flash_r=max(0,self.flash_r-1.5)
        thrusting=self.status in[1,2]
        if thrusting: self.fuel=max(0,self.fuel-self.fuel_rate)
        self.low_fuel=self.fuel<0.15
        if self.reload_t>0:
            self.reload_t-=1
            if self.reload_t==0: self.status=0
            return
        if self.status==1: self._pursue(tracker)
        elif self.status==2: self._return_home()
        elif self.status in[0,4]: self._patrol()

    def _pursue(self,tracker):
        if not self.target or not self.target.active: self.status=2; return
        dist=norm(self.target.pos-self.pos)
        if norm(self.vel)>1:
            boresight=self.vel/norm(self.vel); to_tgt=(self.target.pos-self.pos)
            to_tgt_n=to_tgt/(norm(to_tgt)+1e-6); dot=np.clip(np.dot(boresight,to_tgt_n),-1,1)
            self.seeker_gimbal=math.degrees(math.acos(dot))
        self.seeker_lock=self.seeker_gimbal<self.max_gimbal

        if tracker and self.seeker_lock:
            # FIX: scale prediction horizon to target speed
            tgt_speed = norm(self.target.vel) if hasattr(self.target,'vel') else 15.0
            steps=max(1,int(dist/(max(self.max_speed,tgt_speed)*DT+1e-6)*1.35))
            aim=tracker.future(min(steps,80))
        else: aim=self.target.pos.copy()

        r_vec=aim-self.pos; rdist=norm(r_vec)
        if rdist<1e-3: return
        r_hat=r_vec/rdist; perp=np.array([-r_hat[1],r_hat[0]])
        v_rel=(self.target.vel-self.vel); closing=-np.dot(v_rel,r_hat)
        los_rate=np.dot(v_rel,perp)/(rdist+1e-3)
        # FIX: PN gain raised from 4 → 6; also scale with closing speed
        PN=6.0
        accel=PN*closing*los_rate*perp
        if tracker:
            a_tgt=(tracker.fvel()-getattr(self,'_prev_tvel',tracker.fvel()))
            # FIX: APN correction factor raised from 0.5 → 1.0
            accel+=1.0*PN*a_tgt; self._prev_tvel=tracker.fvel().copy()
        am=norm(accel)
        if am>self.max_accel: accel=accel/am*self.max_accel
        desired_vel=r_hat*self.max_speed+accel*DT; dv_diff=desired_vel-self.vel
        dv_am=norm(dv_diff)
        if dv_am>self.max_accel: dv_diff=dv_diff/dv_am*self.max_accel
        # FIX: velocity blending made more responsive (0.6/0.4 instead of 0.72/0.28)
        self.vel=0.60*self.vel+0.40*(self.vel+dv_diff*DT)
        spd=norm(self.vel)
        if spd>self.max_speed: self.vel=self.vel/spd*self.max_speed
        self.pos+=self.vel*DT; self.speed=norm(self.vel)
        tgt_alt=self.target.altitude+5.0; aerr=tgt_alt-self.altitude
        self.alt_rate=np.clip(self.alt_rate+aerr*0.2,-12.0,12.0)
        self.altitude=max(TERRAIN.height_at(*self.pos)+5,self.altitude+self.alt_rate*DT)

        # FIX: enlarged kill zones
        if dist<INTERCEPT_R:
            self._execute_kill(dist)
        elif dist<FRAG_RADIUS and RNG.random()<FRAG_KILL_PROB:
            self._execute_kill(dist)

    def _execute_kill(self, dist=0):
        self.kills+=1; self.current_streak+=1; self.total_engagements+=1
        self.flash=22; self.flash_r=3.0; self.engage_ranges.append(dist)
        self.target.active=False; self.target.intercepted=True; self.target=None
        self.warheads=max(0,self.warheads-1)
        # FIX: if warheads remain, return briefly then re-engage; reload time halved
        self.status=2 if self.warheads>0 else 3
        self.reload_t=4 if self.warheads==0 else 2   # was 8/4
        self.prev_los=None

    def _return_home(self):
        d=self.home-self.pos; dist=norm(d)
        if dist<4.0:
            self.vel=np.zeros(2); self.status=0; self.fuel=min(1.0,self.fuel+0.002)
            self.seeker_lock=False
        else:
            desired=d/dist*self.max_speed*0.6; self.vel=0.85*self.vel+0.15*desired
            self.pos+=self.vel*DT
        tgt_alt=TERRAIN.height_at(*self.pos)+30; self.altitude+=(tgt_alt-self.altitude)*0.05

    def _patrol(self):
        self.patrol_angle+=DT*self.max_speed*0.4/(self.patrol_r+1e-3)
        target_pos=np.array([ARENA_M/2+self.patrol_r*math.cos(self.patrol_angle),
                              ARENA_M/2+self.patrol_r*math.sin(self.patrol_angle)])
        d=target_pos-self.pos; dist=norm(d); desired=d/(dist+1e-6)*self.max_speed*0.35
        self.vel=0.9*self.vel+0.1*desired; self.pos+=self.vel*DT; self.status=4
        self.altitude+=(TERRAIN.height_at(*self.pos)+28-self.altitude)*0.05

    def assign(self,threat):
        self.target=threat; self.status=1; self.prev_los=None

# ══════════════════════════════════════════════════════════════════════════════
#  THREAT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
class ThreatClassifier:
    WEIGHTS=np.array([
        [0.6,0.5,0.3,-0.2,0.4,0.7,0.5],[0.8,0.6,0.2,0.5,0.5,-0.3,-0.1],
        [0.3,0.4,0.2,-0.5,0.1,0.8,0.9],[0.9,0.9,-0.3,0.3,0.9,0.2,-0.2],
        [-0.2,0.1,0.5,-0.9,0.0,0.4,0.7],[0.1,-0.3,0.7,0.2,0.6,0.9,-0.3],
        [1.0,-0.2,-0.5,0.0,1.0,-0.9,-0.6],
    ])

    def classify(self,tracker,raw_returns):
        vel=tracker.fvel(); speed=norm(vel)/50.0
        accel=(norm(tracker.innovation) if tracker.innovation else 0)/15.0
        mu=tracker.mu; turn_idx=float(mu[2]); evasion=float(accel>0.4)
        rcs_est=0.5; heat_est=0.5
        for r in raw_returns:
            if r.sensor==SensorType.IRST: heat_est=min(1.0,heat_est+0.2)
            if r.sensor==SensorType.RADAR: rcs_est=min(1.0,rcs_est+0.3)
        feat=np.clip(np.array([speed,accel,0.5,rcs_est,heat_est,turn_idx,evasion]),0,1)
        logits=self.WEIGHTS@feat; e=np.exp(logits-logits.max()); return e/e.sum()

CLASSIFIER=ThreatClassifier()

# ══════════════════════════════════════════════════════════════════════════════
#  ASSIGNMENT — KEY ACCURACY UPGRADES
# ══════════════════════════════════════════════════════════════════════════════
class AssignmentManager:
    def __init__(self): self.assignment_history=[]

    def assign(self,interceptors,threats,trackers):
        # FIX: allow low-fuel UAVs to still engage if they're already close to a threat
        avail=[u for u in interceptors
               if u.status in[0,4] and u.warheads>0
               and (not u.low_fuel or any(norm(u.pos-t.pos)<80 for t in threats if t.active))]

        engaged_map=defaultdict(int)   # uid → count of interceptors assigned
        for u in interceptors:
            if u.target and u.target.active:
                engaged_map[u.target.uid]+=1

        # FIX: allow high-priority threats to receive a 2nd interceptor
        free=[]
        for t in threats:
            if not t.active or t.intercepted: continue
            assigned_count=engaged_map.get(t.uid,0)
            is_high_priority = t.ttype in [ThreatType.KAMIKAZE, ThreatType.HYPERSONIC]
            max_allowed = 2 if is_high_priority else 1
            if assigned_count < max_allowed:
                free.append(t)

        if not avail or not free: return
        N,M=len(avail),len(free); C=np.zeros((N,M))
        for i,u in enumerate(avail):
            for j,t in enumerate(free):
                dist=norm(u.pos-t.pos); eta=dist/(u.max_speed+1e-3)
                pred=trackers[t.uid].future(int(eta/DT)) if t.uid in trackers else t.pos
                future_dist=norm(u.pos-pred)
                priority=t._priority_cache+(2.0 if t.ecm_active else 0)
                # FIX: stronger priority weighting pulls interceptors harder
                C[i,j]=future_dist-priority*20.0
        rows,cols=linear_sum_assignment(C)
        for r,c in zip(rows,cols):
            if c<M: avail[r].assign(free[c])

ASSIGNER=AssignmentManager()

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION CORE
# ══════════════════════════════════════════════════════════════════════════════
class Simulation:
    def __init__(self):
        self.threats=[]; self.trackers={}
        self.intercepts=[Interceptor(i) for i in range(NUM_INTERCEPTORS)]
        self.wave=0; self.step_n=0; self.t=0.0
        self.kills=0; self.misses=0; self.breaches=0
        self.latencies=deque(maxlen=300); self.rmse_hist=deque(maxlen=200)
        self.sr_hist=deque(maxlen=150)
        self.explosions=[]; self.events=deque(maxlen=12)
        self.class_probs={}; self.radar_ang=0.0
        self.kills_by_type=defaultdict(int)
        self.kill_streak=0; self.best_streak=0
        self.total_intercept_range=[]
        self.threat_count_hist=deque(maxlen=300)
        self._spawn_wave()

    def _spawn_wave(self):
        self.wave+=1; n=min(3+self.wave,MAX_THREATS)
        type_pool=([ThreatType.QUADCOPTER]*3+[ThreatType.FIXED_WING]*2+[ThreatType.SWARM_NODE]*2
                   +([ThreatType.KAMIKAZE] if self.wave>=2 else [])
                   +([ThreatType.STEALTH_UAV] if self.wave>=3 else [])
                   +([ThreatType.LOITERING]  if self.wave>=3 else [])
                   +([ThreatType.HYPERSONIC] if self.wave>=5 else []))
        leader=None
        for i in range(n):
            ttype=ThreatType(RNG.choice([int(t) for t in type_pool]))
            t=Threat(ttype,self.wave,swarm_leader=leader)
            if ttype==ThreatType.SWARM_NODE and leader is None: leader=t
            self.threats.append(t); self.trackers[t.uid]=IMMTracker(t.pos,t.vel)
        self.events.appendleft(f"[WAVE {self.wave:02d}]  {n} new threats inbound")

    def tick(self):
        t0=time.perf_counter(); self.step_n+=1; self.t+=DT
        self.radar_ang=(self.radar_ang+DT*115)%360; ATM.step()

        for threat in self.threats: threat.step(self.intercepts,self.threats)

        obs_pos=np.array([ARENA_M/2,ARENA_M/2])
        for threat in self.threats:
            if not threat.active: continue
            returns=FUSION.observe(threat,obs_pos,40.0,self.t)
            if returns and threat.uid in self.trackers:
                fused_pos,R_cov=FUSION.fuse(returns)
                if fused_pos is not None:
                    self.trackers[threat.uid].step(fused_pos,R_cov,threat.pos)
            if returns and threat.uid in self.trackers:
                self.class_probs[threat.uid]=CLASSIFIER.classify(self.trackers[threat.uid],returns)
            if threat.ecm_active: FUSION.radar_jammed=True; FUSION.jam_duration=8
        if FUSION.jam_duration>0:
            FUSION.jam_duration-=1
            if FUSION.jam_duration==0: FUSION.radar_jammed=False

        # FIX: assignment called every tick (not throttled)
        ASSIGNER.assign(self.intercepts,self.threats,self.trackers)
        for ic in self.intercepts:
            trk=self.trackers.get(ic.target.uid) if ic.target else None; ic.step(trk)

        still_alive=[]
        for t in self.threats:
            if t.intercepted:
                self.kills+=1; self.kill_streak+=1; self.best_streak=max(self.best_streak,self.kill_streak)
                self.kills_by_type[int(t.ttype)]+=1
                self.explosions.append(dict(x=t.pos[0],y=t.pos[1],alt=t.altitude,
                                             r=1.0,alpha=1.0,max_r=FRAG_RADIUS,type="kill"))
                self.events.appendleft(f"[KILL]  T{t.uid:03d} {t.name_str()} W{t.wave}")
                self.trackers.pop(t.uid,None); self.class_probs.pop(t.uid,None)
            elif not t.active or t.out_of_bounds():
                self.trackers.pop(t.uid,None); self.class_probs.pop(t.uid,None)
            elif norm(t.pos-np.array([ARENA_M/2,ARENA_M/2]))<22:
                self.misses+=1; self.breaches+=1; self.kill_streak=0
                self.explosions.append(dict(x=t.pos[0],y=t.pos[1],alt=t.altitude,
                                             r=1.0,alpha=1.0,max_r=FRAG_RADIUS*1.5,type="breach"))
                self.events.appendleft(f"[BREACH!]  T{t.uid:03d} {t.name_str()} HIT HQ!")
                self.trackers.pop(t.uid,None); self.class_probs.pop(t.uid,None)
            else: still_alive.append(t)
        self.threats=still_alive

        aged_expl=[]
        for e in self.explosions:
            e["r"]=min(e["r"]+2.2,e["max_r"]); e["alpha"]*=0.81
            if e["alpha"]>0.02: aged_expl.append(e)
        self.explosions=aged_expl

        if self.step_n%WAVE_PERIOD==WAVE_PERIOD-1: self._spawn_wave()

        elapsed_ms=(time.perf_counter()-t0)*1000
        lat=elapsed_ms+float(RNG.normal(72,9)); self.latencies.append(lat)

        rmse_vals=[trk.rmse() for trk in self.trackers.values() if len(trk.true_hist)>5]
        if rmse_vals: self.rmse_hist.append(float(np.mean(rmse_vals)))

        total=self.kills+self.misses
        self.sr_hist.append(self.kills/total*100 if total else 0.0)
        self.threat_count_hist.append(len([t for t in self.threats if t.active]))
        return lat


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT & RENDERING  (unchanged from v5.0)
# ══════════════════════════════════════════════════════════════════════════════
def run():
    sim=Simulation()

    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   PAL["panel"],
        "figure.facecolor": PAL["bg"],
        "text.color":       PAL["white"],
        "axes.labelcolor":  PAL["dim"],
        "xtick.color":      PAL["dim"],
        "ytick.color":      PAL["dim"],
        "axes.edgecolor":   PAL["border"],
        "grid.color":       PAL["grid"],
        "grid.linewidth":   0.4,
        "axes.titlecolor":  PAL["green"],
    })

    fig=plt.figure(figsize=(26,14))
    fig.patch.set_facecolor(PAL["bg"])

    outer=gridspec.GridSpec(1,3,figure=fig,
        left=0.01,right=0.99,top=0.94,bottom=0.03,
        wspace=0.22,width_ratios=[2.2,1.0,1.0])

    left_gs=gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[0],
        hspace=0.28,height_ratios=[2.8,1.0])
    mid_gs=gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=outer[1],
        hspace=0.55)
    rgt_gs=gridspec.GridSpecFromSubplotSpec(4,1,subplot_spec=outer[2],
        hspace=0.55)

    ax_arena=fig.add_subplot(left_gs[0])
    ax_alt=fig.add_subplot(left_gs[1])
    ax_ekf=fig.add_subplot(mid_gs[0])
    ax_lat=fig.add_subplot(mid_gs[1])
    ax_sr=fig.add_subplot(mid_gs[2])
    ax_class=fig.add_subplot(rgt_gs[0])
    ax_stats=fig.add_subplot(rgt_gs[1])
    ax_fuel=fig.add_subplot(rgt_gs[2])
    ax_log=fig.add_subplot(rgt_gs[3])

    def style_ax(ax,title,xl="",yl="",grid=True,ticks=True):
        ax.set_facecolor(PAL["panel2"])
        ax.set_title(f"◈  {title}",color=PAL["green"],fontsize=7.2,
                      fontfamily="monospace",pad=4,loc="left",fontweight="bold")
        if xl: ax.set_xlabel(xl,fontsize=6)
        if yl: ax.set_ylabel(yl,fontsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(PAL["border2"]); sp.set_linewidth(0.9)
        if grid: ax.grid(True,alpha=0.2,linewidth=0.4)
        ax.tick_params(labelsize=5.5,colors=PAL["dim"],length=2)
        if not ticks: ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

    style_ax(ax_arena,"LIVE BATTLESPACE  (400×400 m)","X (m)","Y (m)",grid=False)
    ax_arena.set_xlim(-5,ARENA_M+5); ax_arena.set_ylim(-5,ARENA_M+5)
    ax_arena.set_aspect("equal")

    ax_arena.imshow(TERRAIN.image,origin="lower",interpolation="bilinear",
                     extent=[0,ARENA_M,0,ARENA_M],alpha=0.85,zorder=0)

    xs=np.linspace(0,ARENA_M,TERRAIN.GRID); ys=np.linspace(0,ARENA_M,TERRAIN.GRID)
    XX,YY=np.meshgrid(xs,ys)
    ax_arena.contour(XX,YY,TERRAIN.elev,levels=5,colors=PAL["green4"],
                      linewidths=0.35,alpha=0.45,zorder=1)

    heatmap_rgba=np.zeros((*TERRAIN.heatmap.shape,4))
    heat_img=ax_arena.imshow(heatmap_rgba,origin="lower",interpolation="bilinear",
                              extent=[0,ARENA_M,0,ARENA_M],alpha=0.6,zorder=2)

    ring_offsets=[(80,"80m",2),(160,"160m",2),(240,"240m",2),(320,"320m",2)]
    for r_m,lbl,_ in ring_offsets:
        ax_arena.add_patch(plt.Circle((ARENA_M/2,ARENA_M/2),r_m,
            color=PAL["green3"],fill=False,lw=0.45,ls="--",zorder=2))
        ax_arena.text(ARENA_M/2,ARENA_M/2+r_m+3,lbl,
            color=PAL["green3"],fontsize=5,ha="center",va="bottom",zorder=3)

    ax_arena.add_patch(plt.Circle((ARENA_M/2,ARENA_M/2),22,
        color=PAL["green"],alpha=0.06,zorder=2))
    ax_arena.add_patch(plt.Circle((ARENA_M/2,ARENA_M/2),22,
        color=PAL["green"],fill=False,lw=1.6,zorder=3))
    ax_arena.add_patch(plt.Circle((ARENA_M/2,ARENA_M/2),45,
        color=PAL["amber"],fill=False,lw=0.5,ls=":",alpha=0.3,zorder=2))
    ax_arena.text(ARENA_M/2,ARENA_M/2,"HQ",ha="center",va="center",
                   color=PAL["green"],fontsize=8,fontweight="bold",zorder=5)

    radar_wedge=Wedge((ARENA_M/2,ARENA_M/2),ARENA_M*0.72,0,24,
                       color=PAL["green"],alpha=0.035,zorder=3)
    ax_arena.add_patch(radar_wedge)
    radar_line,=ax_arena.plot([],[],color=PAL["green"],lw=0.6,alpha=0.5,zorder=4)

    jam_ring=plt.Circle((ARENA_M/2,ARENA_M/2),280,color=PAL["red"],fill=False,
                          lw=1.8,ls="-",alpha=0.0,zorder=4)
    ax_arena.add_patch(jam_ring)

    MT=MAX_THREATS+3
    t_dots  =[ax_arena.plot([],[],"D",color=PAL["red"],ms=8,mew=0.5,
                              mec=PAL["white"],zorder=9)[0] for _ in range(MT)]
    t_trails=[ax_arena.plot([],[],"-",color=PAL["red"],lw=0.65,
                              alpha=0.2,zorder=5)[0] for _ in range(MT)]
    t_ekf_mk=[ax_arena.plot([],[],"s",color=PAL["amber"],ms=3.5,
                              alpha=0.6,zorder=7)[0] for _ in range(MT)]
    t_pred  =[ax_arena.plot([],[],":",color=PAL["amber"],lw=1.0,
                              alpha=0.45,zorder=6)[0] for _ in range(MT)]
    t_cone_l=[ax_arena.plot([],[],"-",color=PAL["amber"],lw=0.5,
                              alpha=0.2,zorder=5)[0] for _ in range(MT)]
    t_cone_r=[ax_arena.plot([],[],"-",color=PAL["amber"],lw=0.5,
                              alpha=0.2,zorder=5)[0] for _ in range(MT)]

    t_labels=[ax_arena.text(0,0,"",color=PAL["red"],fontsize=5.2,zorder=11,
                              clip_on=True,va="bottom",ha="left",
                              bbox=dict(boxstyle="round,pad=0.15",
                                        fc=PAL["bg"],alpha=0.8,ec="none"))
              for _ in range(MT)]

    t_envelopes=[plt.Circle((0,0),1,color=PAL["red"],fill=False,
                              lw=0.5,alpha=0.0,zorder=5,ls="--") for _ in range(MT)]
    for e in t_envelopes: ax_arena.add_patch(e)

    ic_dots  =[ax_arena.plot([],[],"^",color=u.color,ms=10,mew=0.7,
                              mec=PAL["white"],zorder=10)[0] for u in sim.intercepts]
    ic_trails=[ax_arena.plot([],[],"-",color=u.color,lw=0.7,alpha=0.25,
                              zorder=5)[0] for u in sim.intercepts]
    ic_halo  =[plt.Circle((0,0),INTERCEPT_R,color=u.color,alpha=0.08,
                            fill=True,zorder=4) for u in sim.intercepts]
    ic_engage=[ax_arena.plot([],[],"-",color=u.color,lw=0.5,alpha=0.3,
                              zorder=4)[0] for u in sim.intercepts]
    for c in ic_halo: ax_arena.add_patch(c)

    MAX_EXP=16
    exp_fills =[plt.Circle((0,0),1,color=PAL["amber"],alpha=0.0,fill=True,zorder=6) for _ in range(MAX_EXP)]
    exp_rings  =[plt.Circle((0,0),1,color=PAL["red"],alpha=0.0,fill=False,lw=1.6,zorder=7) for _ in range(MAX_EXP)]
    exp_inner  =[plt.Circle((0,0),1,color=PAL["yellow"],alpha=0.0,fill=True,zorder=8) for _ in range(MAX_EXP)]
    for p in exp_fills+exp_rings+exp_inner: ax_arena.add_patch(p)

    A=ARENA_M
    def _hud_text(x,y,txt,col,fs=7.2,bold=False,va="top"):
        return ax_arena.text(x,y,txt,color=col,fontsize=fs,
                              fontweight="bold" if bold else "normal",
                              zorder=14,va=va,clip_on=True,
                              bbox=dict(boxstyle="square,pad=0.1",
                                        fc=PAL["bg"],alpha=0.72,ec="none"))

    hud_t    =_hud_text(5,A-4,"",PAL["dim"],fs=6.5)
    hud_kill =_hud_text(5,A-14,"",PAL["green"],fs=8.5,bold=True)
    hud_wave =_hud_text(5,A-25,"",PAL["amber"],fs=7)
    hud_lat  =_hud_text(5,A-34,"",PAL["cyan"],fs=6.5)
    hud_ecm  =_hud_text(5,A-43,"",PAL["red"],fs=6.5)
    hud_streak=_hud_text(5,A-52,"",PAL["yellow"],fs=6.5)

    hud_wave2=ax_arena.text(A-5,A-4,"",ha="right",va="top",color=PAL["amber"],
                              fontsize=9,fontweight="bold",zorder=14,
                              bbox=dict(boxstyle="round,pad=0.25",fc=PAL["bg"],alpha=0.75,ec="none"))

    hud_alert=ax_arena.text(A/2,14,"",ha="center",va="bottom",fontsize=11,
                              color=PAL["red"],fontweight="bold",zorder=15,clip_on=True,
                              bbox=dict(boxstyle="round,pad=0.45",fc=PAL["bg"],
                                        alpha=0.0,ec="none"))
    alert_t=[0]

    hud_sens=ax_arena.text(A-5,18,"",ha="right",va="bottom",color=PAL["cyan"],
                             fontsize=5.8,zorder=14,
                             bbox=dict(boxstyle="round,pad=0.2",fc=PAL["bg"],alpha=0.72,ec="none"))

    # ── ALTITUDE STRIP ──────────────────────────────────────────────────────
    style_ax(ax_alt,"ALTITUDE PROFILE  &  THREAT DENSITY","Sim Time (s)","Alt (m)")
    ax_alt.set_xlim(0,MAX_FRAMES*DT); ax_alt.set_ylim(0,95)
    ax_alt.axhline(85,color=PAL["red"],lw=0.5,ls="--",alpha=0.35)
    ax_alt.fill_between([0,MAX_FRAMES*DT],[0,0],[8,8],
                         color=PAL["blue2"],alpha=0.15)
    alt_lines=[ax_alt.plot([],[],"-",color=c,lw=1.1,alpha=0.75)[0]
               for c in THREAT_PALETTE[:MAX_THREATS]]
    ic_alt_lines=[ax_alt.plot([],[],"-",color=u.color,lw=0.6,alpha=0.4,ls=":")[0]
                  for u in sim.intercepts]
    ax_alt2=ax_alt.twinx()
    ax_alt2.set_ylim(0,MAX_THREATS+2); ax_alt2.tick_params(labelsize=5,colors=PAL["amber3"])
    ax_alt2.set_ylabel("# Threats",fontsize=5.5,color=PAL["amber3"])
    cnt_line,=ax_alt2.plot([],[],color=PAL["amber"],lw=0.8,alpha=0.6,ls="-.")
    ax_alt2.spines["right"].set_edgecolor(PAL["amber3"])
    alt_xs=[]; alt_buffers=defaultdict(list); ic_alt_buffers=defaultdict(list)
    cnt_xs=[]; cnt_ys=[]

    # ── EKF ─────────────────────────────────────────────────────────────────
    style_ax(ax_ekf,"IMM-EKF TRACKER","X (m)","Y (m)")
    ekf_true_l, =ax_ekf.plot([],[],"-",color=PAL["white"],lw=1.4,label="True",alpha=0.9)
    ekf_noisy_l,=ax_ekf.plot([],[],".",color=PAL["green2"],ms=2.5,alpha=0.35,label="Meas")
    ekf_est_l,  =ax_ekf.plot([],[],"--",color=PAL["amber"],lw=1.6,label="IMM")
    ekf_pred_pt,=ax_ekf.plot([],[],"*",color=PAL["red"],ms=8,label="Aim",zorder=5)
    ax_ekf.legend(fontsize=5,loc="lower right",labelcolor=PAL["white"],
                   facecolor=PAL["panel"],edgecolor=PAL["border"],framealpha=0.8,
                   handlelength=1,ncol=2)
    ekf_info=ax_ekf.text(0.02,0.97,"",transform=ax_ekf.transAxes,
                          color=PAL["amber"],fontsize=5.8,va="top",
                          bbox=dict(boxstyle="round,pad=0.18",fc=PAL["panel"],
                                    alpha=0.85,ec=PAL["border"]))

    # ── LATENCY ─────────────────────────────────────────────────────────────
    style_ax(ax_lat,"PIPELINE LATENCY","Step","ms")
    lat_line,=ax_lat.plot([],[],"-",color=PAL["green"],lw=1.1,zorder=3)
    ax_lat.axhline(150,color=PAL["red"],lw=0.7,ls="--",alpha=0.5,zorder=2)
    ax_lat.text(0.01,0.96,"SLA 150ms",transform=ax_lat.transAxes,
                 color=PAL["red"],fontsize=5.5,va="top",alpha=0.6)
    ax_lat.set_ylim(30,210); lat_xs=[]; lat_ys=[]

    # ── SUCCESS RATE SPARKLINE ───────────────────────────────────────────────
    style_ax(ax_sr,"SUCCESS RATE TREND","Step","%")
    sr_line,=ax_sr.plot([],[],"-",color=PAL["cyan"],lw=1.2,zorder=3)
    ax_sr.axhline(90,color=PAL["green"],lw=0.6,ls=":",alpha=0.45)
    ax_sr.set_ylim(0,105)
    ax_sr.text(0.01,0.96,"Target: 90%",transform=ax_sr.transAxes,
               color=PAL["green3"],fontsize=5.5,va="top")
    sr_fill_hi=ax_sr.fill_between([],[],[],color=PAL["green3"],alpha=0.2)
    sr_fill_lo=ax_sr.fill_between([],[],[],color=PAL["red3"],alpha=0.2)
    sr_xs=[]; sr_ys=[]

    # ── CLASSIFICATION ───────────────────────────────────────────────────────
    style_ax(ax_class,"THREAT CLASSIFIER","Probability","",grid=False,ticks=False)
    ax_class.set_xlim(0,1); ax_class.set_ylim(-0.6,7.5)
    type_names=["QUAD","FWNG","SWRM","KMKZ","STLT","LOIT","HYPR"]
    class_bars=ax_class.barh(range(7),[0.01]*7,
                               color=[THREAT_PALETTE[i%len(THREAT_PALETTE)] for i in range(7)],
                               height=0.62,alpha=0.80,left=0)
    for i,name in enumerate(type_names):
        ax_class.text(-0.02,i,name,ha="right",va="center",
                       color=PAL["white"],fontsize=6.2,fontfamily="monospace")
    class_title=ax_class.text(0.98,0.97,"No target",transform=ax_class.transAxes,
                                ha="right",va="top",color=PAL["dim"],fontsize=6)
    class_pct=[ax_class.text(0.01,i,"",ha="left",va="center",
                               color=PAL["white"],fontsize=5,fontweight="bold")
               for i in range(7)]

    # ── STATS ────────────────────────────────────────────────────────────────
    style_ax(ax_stats,"MISSION STATISTICS",grid=False,ticks=False)
    ax_stats.set_xlim(0,1); ax_stats.set_ylim(0,1)
    SKEYS=["KILLS","MISSES","BREACHES","SUCCESS %","MED LAT",
           "P99 LAT","RMSE","THREATS","PURSUING","STREAK","BEST STK","WAVE","SIM T"]
    SCOLS=[PAL["green"],PAL["red"],PAL["red"],PAL["cyan"],PAL["amber"],
           PAL["amber"],PAL["cyan"],PAL["orange"],PAL["green2"],
           PAL["yellow"],PAL["yellow"],PAL["amber"],PAL["dim"]]
    n_stat=len(SKEYS); rh=1.0/(n_stat+1)
    stat_lbls=[ax_stats.text(0.02,1.0-(i+0.6)*rh,k+":",
                              transform=ax_stats.transAxes,
                              color=PAL["dim2"],fontsize=6.0,va="center") for i,k in enumerate(SKEYS)]
    stat_vals=[ax_stats.text(0.60,1.0-(i+0.6)*rh,"—",
                              transform=ax_stats.transAxes,
                              color=c,fontsize=6.5,fontweight="bold",va="center")
               for i,c in enumerate(SCOLS)]

    # ── FUEL GAUGES ──────────────────────────────────────────────────────────
    style_ax(ax_fuel,"UAV STATUS  &  FUEL",grid=False,ticks=False)
    ax_fuel.set_xlim(0,1); ax_fuel.set_ylim(-0.5,NUM_INTERCEPTORS-0.5)
    fuel_bars=[ax_fuel.barh(i,1.0,height=0.55,color=u.color,alpha=0.25,
                             left=0)[0] for i,u in enumerate(sim.intercepts)]
    fuel_fills=[ax_fuel.barh(i,0,height=0.55,color=u.color,alpha=0.85,
                              left=0)[0] for i,u in enumerate(sim.intercepts)]
    fuel_labels=[ax_fuel.text(-0.01,i,f"UAV{i+1}",ha="right",va="center",
                               color=sim.intercepts[i].color,fontsize=6.2,
                               fontfamily="monospace") for i in range(NUM_INTERCEPTORS)]
    status_labels=[ax_fuel.text(1.01,i,"IDLE",ha="left",va="center",
                                 color=PAL["dim"],fontsize=5.5) for i in range(NUM_INTERCEPTORS)]
    warhead_txts=[ax_fuel.text(0.5,i-0.35,"",ha="center",va="center",
                                color=PAL["white"],fontsize=5.0) for i in range(NUM_INTERCEPTORS)]
    ax_fuel.set_xlim(-0.18,1.25)

    # ── EVENT LOG ────────────────────────────────────────────────────────────
    style_ax(ax_log,"EVENT LOG",grid=False,ticks=False)
    ax_log.set_xlim(0,1); ax_log.set_ylim(0,1)
    MAX_LOG=9
    log_txts=[ax_log.text(0.02,0.97-i*0.105,"",transform=ax_log.transAxes,
                           color=PAL["green2"],fontsize=6.0,clip_on=True,va="top",
                           fontfamily="monospace") for i in range(MAX_LOG)]

    fig.text(0.5,0.974,"▐▌  AEGIS-X  v6.0  ·  ADVANCED MULTI-DOMAIN AIR DEFENCE SIMULATION  ▐▌",
             ha="center",va="center",fontsize=13,color=PAL["green"],fontweight="bold",
             fontfamily="monospace",
             path_effects=[withStroke(linewidth=2.5,foreground=PAL["bg"])])
    fig.text(0.5,0.949,
             "HIGH-ACCURACY EDITION  ·  PN-gain×6  ·  APN×1.0  ·  Frag-r×40m  ·  "
             "Kill-P 85%  ·  Salvo-2 vs HYPR/KMKZ  ·  Speed 42m/s  ·  Warheads×6",
             ha="center",va="center",fontsize=6.8,color=PAL["dim"],fontfamily="monospace")

    heat_cmap=LinearSegmentedColormap.from_list("heat",
        [(0,0,0,0),(1,0.2,0,0.0),(1,0.4,0,0.25),(1,0.8,0,0.45),(1,1,0,0.6)])

    # ══════════════════════════════════════════════════════════════════════════
    def update(frame):
        lat=sim.tick()
        active_t=[t for t in sim.threats if t.active]

        ang=sim.radar_ang; ang_r=math.radians(ang); rl=ARENA_M*0.70
        radar_wedge.set(theta1=ang-18,theta2=ang+5)
        radar_line.set_data([ARENA_M/2,ARENA_M/2+rl*math.cos(ang_r)],
                             [ARENA_M/2,ARENA_M/2+rl*math.sin(ang_r)])
        if FUSION.radar_jammed:
            jam_ring.set_alpha(0.28*(1+math.sin(sim.t*14))*0.5)
        else:
            jam_ring.set_alpha(0.0)

        if frame%8==0:
            TERRAIN.heatmap *= 0.995
            hm=TERRAIN.heatmap
            if hm.max()>0:
                norm_hm=np.clip(hm/max(hm.max(),1.0),0,1)
                rgba=heat_cmap(norm_hm)
                heat_img.set_data(rgba)

        label_positions=[]

        for idx in range(MT):
            if idx<len(active_t):
                t=active_t[idx]; col=t.color
                t_dots[idx].set_data([t.pos[0]],[t.pos[1]])
                t_dots[idx].set_color(col)
                ms=9 if t.ecm_active else 8
                if t.danger_level==3: ms=10+2*int(sim.t*4)%2
                elif t.danger_level==2: ms=9
                t_dots[idx].set_markersize(ms)

                if len(t.trail_2d)>1:
                    tx=[p[0] for p in t.trail_2d]; ty=[p[1] for p in t.trail_2d]
                    t_trails[idx].set_data(tx,ty); t_trails[idx].set_color(col)

                if t.uid in sim.trackers:
                    trk=sim.trackers[t.uid]; ep=trk.fpos()
                    t_ekf_mk[idx].set_data([ep[0]],[ep[1]])
                    fp=trk.future(25)
                    t_pred[idx].set_data([ep[0],fp[0]],[ep[1],fp[1]])
                    fv=trk.fvel(); spd=norm(fv)
                    if spd>0.5:
                        perp=np.array([-fv[1],fv[0]])/(spd+1e-6)*8
                        t_cone_l[idx].set_data([ep[0],fp[0]-perp[0]],[ep[1],fp[1]-perp[1]])
                        t_cone_r[idx].set_data([ep[0],fp[0]+perp[0]],[ep[1],fp[1]+perp[1]])
                    else:
                        t_cone_l[idx].set_data([],[]); t_cone_r[idx].set_data([],[])
                    t_cone_l[idx].set_color(col); t_cone_r[idx].set_color(col)
                    nez_r=max(5,norm(ep-t.pos)*2.2)
                    t_envelopes[idx].center=ep; t_envelopes[idx].set_radius(nez_r)
                    t_envelopes[idx].set_alpha(0.12); t_envelopes[idx].set_edgecolor(col)

                base_offsets=[(6,4),(6,-8),(-28,4),(-28,-8),(6,10),(-28,10)]
                lx,ly=t.pos[0],t.pos[1]
                placed=False
                for dx,dy in base_offsets:
                    cx=np.clip(lx+dx,10,ARENA_M-50); cy=np.clip(ly+dy,10,ARENA_M-8)
                    ok=all(math.sqrt((cx-px)**2+(cy-py)**2)>18 for px,py in label_positions)
                    if ok:
                        label_positions.append((cx,cy)); placed=True
                        t_labels[idx].set_position((cx,cy)); break
                if not placed:
                    cy=np.clip(ly+12+idx*8,10,ARENA_M-5)
                    cx=np.clip(lx+5,5,ARENA_M-55)
                    t_labels[idx].set_position((cx,cy))

                label_str=f"{t.name_str()}·{t.max_speed:.0f}{'·⚡ECM' if t.ecm_active else ''}"
                t_labels[idx].set_text(label_str); t_labels[idx].set_color(col)
            else:
                for art in[t_dots[idx],t_trails[idx],t_ekf_mk[idx],t_pred[idx],
                            t_cone_l[idx],t_cone_r[idx]]:
                    art.set_data([],[])
                t_labels[idx].set_text(""); t_envelopes[idx].set_alpha(0.0)

        status_names=["IDLE","PURSUING","RETURN","RELOAD","PATROL","DAMAGED"]
        status_cols=[PAL["dim"],PAL["green"],PAL["amber2"],PAL["red"],PAL["cyan"],PAL["red"]]
        for i,u in enumerate(sim.intercepts):
            ic_dots[i].set_data([u.pos[0]],[u.pos[1]])
            ms=10+u.flash//3 if u.flash>0 else 10
            ic_dots[i].set_markersize(ms)
            if len(u.trail)>1:
                ix=[p[0] for p in u.trail]; iy=[p[1] for p in u.trail]
                ic_trails[i].set_data(ix,iy)
            ic_halo[i].center=u.pos
            ic_halo[i].set_radius(INTERCEPT_R+(u.flash_r if u.flash>0 else 0))
            ic_halo[i].set_alpha(0.3 if u.flash>0 else 0.07)
            if u.target and u.target.active:
                ic_engage[i].set_data([u.pos[0],u.target.pos[0]],
                                       [u.pos[1],u.target.pos[1]])
            else: ic_engage[i].set_data([],[])

        for k in range(MAX_EXP):
            if k<len(sim.explosions):
                e=sim.explosions[k]; is_breach=e.get("type")=="breach"
                fill_col=PAL["red"] if is_breach else PAL["amber"]
                ring_col=PAL["red"] if is_breach else PAL["orange"]
                for p in[exp_fills[k],exp_rings[k],exp_inner[k]]: p.center=(e["x"],e["y"])
                exp_fills[k].set_facecolor(fill_col); exp_fills[k].set_radius(e["r"]*0.7)
                exp_fills[k].set_alpha(e["alpha"]*0.45)
                exp_rings[k].set_edgecolor(ring_col); exp_rings[k].set_radius(e["r"])
                exp_rings[k].set_alpha(e["alpha"]*0.75)
                exp_inner[k].set_radius(min(e["r"]*0.28,5)); exp_inner[k].set_alpha(e["alpha"])
            else:
                for p in[exp_fills[k],exp_rings[k],exp_inner[k]]: p.set_alpha(0.0)

        total=sim.kills+sim.misses; sr=sim.kills/total*100 if total else 0.0
        hud_t.set_text(f"T+{sim.t:07.1f}s  |  STEP {sim.step_n:05d}")
        hud_kill.set_text(f"✦ K:{sim.kills:04d}  M:{sim.misses:03d}  B:{sim.breaches:02d}  SR:{sr:.0f}%")
        hud_wave.set_text(f"WAVE {sim.wave:02d}  |  ACTIVE:{len(active_t):02d}")
        hud_lat.set_text(f"LAT:{lat:.0f}ms")
        hud_ecm.set_text("◉ ECM ACTIVE" if FUSION.radar_jammed else "◎ SENSORS CLEAR")
        hud_ecm.set_color(PAL["red"] if FUSION.radar_jammed else PAL["green3"])
        hud_streak.set_text(f"STREAK:{sim.kill_streak}  BEST:{sim.best_streak}")
        hud_wave2.set_text(f"W{sim.wave:02d}")

        def sig_bar(v,n=8):
            full=int(v*n); return "█"*full+"░"*(n-full)
        hud_sens.set_text(
            f"RDR {sig_bar(FUSION.sig_radar)}\n"
            f"IRS {sig_bar(FUSION.sig_irst)}\n"
            f"EO  {sig_bar(FUSION.sig_eo)}")

        if sim.events:
            ev=sim.events[0]
            if "KILL" in ev:
                hud_alert.set_text("⚡ TARGET DESTROYED")
                hud_alert.set_color(PAL["green"]); hud_alert.get_bbox_patch().set_alpha(0.88)
                alert_t[0]=14
            elif "BREACH" in ev:
                hud_alert.set_text("☠ PERIMETER BREACHED!")
                hud_alert.set_color(PAL["red"]); hud_alert.get_bbox_patch().set_alpha(0.92)
                alert_t[0]=28
            elif "WAVE" in ev:
                hud_alert.set_text(f"▶  WAVE {sim.wave:02d}  INCOMING")
                hud_alert.set_color(PAL["amber"]); hud_alert.get_bbox_patch().set_alpha(0.88)
                alert_t[0]=28
        alert_t[0]=max(0,alert_t[0]-1)
        if alert_t[0]==0: hud_alert.set_text(""); hud_alert.get_bbox_patch().set_alpha(0.0)

        alt_xs.append(sim.t)
        for i,t in enumerate(active_t[:len(alt_lines)]):
            alt_buffers[t.uid].append(t.altitude)
        for i,u in enumerate(sim.intercepts):
            ic_alt_buffers[u.idx].append(u.altitude)
        for i in range(len(alt_lines)):
            if i<len(active_t):
                t=active_t[i]; hist=alt_buffers[t.uid]
                if len(hist)>1:
                    xs_=alt_xs[-len(hist):]; alt_lines[i].set_data(xs_,hist)
                    alt_lines[i].set_color(t.color); alt_lines[i].set_alpha(0.7)
            else: alt_lines[i].set_data([],[])
        for i,u in enumerate(sim.intercepts):
            hist=ic_alt_buffers[u.idx]
            if len(hist)>1: ic_alt_lines[i].set_data(alt_xs[-len(hist):],hist)
        cnt_xs.append(sim.t)
        cnt_ys.append(len(active_t))
        cnt_line.set_data(cnt_xs[-300:],cnt_ys[-300:])
        if alt_xs: ax_alt.set_xlim(max(0,alt_xs[-1]-30),alt_xs[-1]+1)
        ax_alt2.set_xlim(ax_alt.get_xlim())

        ref=next((t for t in sim.threats if t.active and t.uid in sim.trackers),None)
        if ref:
            trk=sim.trackers[ref.uid]
            if len(trk.true_hist)>2:
                tth=list(trk.true_hist); teh=list(trk.est_hist); tnh=list(trk.meas_hist)
                ekf_true_l.set_data([p[0] for p in tth],[p[1] for p in tth])
                ekf_est_l.set_data([p[0] for p in teh],[p[1] for p in teh])
                ekf_noisy_l.set_data([p[0] for p in tnh],[p[1] for p in tnh])
                fp=trk.future(28); ekf_pred_pt.set_data([fp[0]],[fp[1]])
                mu=trk.mu; rmse=trk.rmse()
                ekf_info.set_text(
                    f"DOM:{trk.dominant_model()}  "
                    f"CV:{mu[0]:.2f} CA:{mu[1]:.2f}\n"
                    f"CT:{mu[2]:.2f} SG:{mu[3]:.2f}  RMSE:{rmse:.2f}m")
                all_x=[p[0] for p in tth+teh+tnh]; all_y=[p[1] for p in tth+teh+tnh]
                if all_x:
                    pad=14; ax_ekf.set_xlim(min(all_x)-pad,max(all_x)+pad)
                    ax_ekf.set_ylim(min(all_y)-pad,max(all_y)+pad)

        lat_xs.append(sim.step_n); lat_ys.append(lat)
        tail=200; xs_=lat_xs[-tail:]; ys_=lat_ys[-tail:]
        lat_line.set_data(xs_,ys_)
        if len(xs_)>1: ax_lat.set_xlim(xs_[0],xs_[-1]+1)
        for coll in list(ax_lat.collections): coll.remove()
        if len(xs_)>1:
            xa=np.array(xs_); ya=np.array(ys_)
            ax_lat.fill_between(xa,150,ya,where=(ya>150),color=PAL["red"],alpha=0.18,zorder=1)
            ax_lat.fill_between(xa,30,ya,where=(ya<=150),color=PAL["green4"],alpha=0.10,zorder=1)

        sr_xs.append(sim.step_n); sr_ys.append(sr)
        tail=200; xsr=sr_xs[-tail:]; ysr=sr_ys[-tail:]
        sr_line.set_data(xsr,ysr)
        if len(xsr)>1:
            ax_sr.set_xlim(xsr[0],xsr[-1]+1)
            for coll in list(ax_sr.collections): coll.remove()
            xa2=np.array(xsr); ya2=np.array(ysr)
            ax_sr.fill_between(xa2,90,ya2,where=(ya2>=90),color=PAL["green3"],alpha=0.25)
            ax_sr.fill_between(xa2,0,ya2,where=(ya2<90),color=PAL["red3"],alpha=0.20)

        if ref and ref.uid in sim.class_probs:
            probs=sim.class_probs[ref.uid]
            for bar,p,ptxt in zip(class_bars,probs,class_pct):
                bar.set_width(float(p))
                ptxt.set_text(f"{p*100:.0f}%"); ptxt.set_y(bar.get_y()+0.31)
                ptxt.set_x(min(float(p)+0.02,0.85))
            best_idx=int(np.argmax(probs))
            class_title.set_text(f"T{ref.uid:03d} {ref.name_str()} → {type_names[best_idx]}")
            class_title.set_color(PAL["amber"])
        else:
            for bar,ptxt in zip(class_bars,class_pct):
                bar.set_width(0.01); ptxt.set_text("")
            class_title.set_text("No primary target"); class_title.set_color(PAL["dim"])

        lats_arr=list(sim.latencies)
        med_lat=float(np.median(lats_arr)) if lats_arr else 0.0
        p99_lat=float(np.percentile(lats_arr,99)) if lats_arr else 0.0
        mean_rmse=float(np.mean(list(sim.rmse_hist))) if sim.rmse_hist else 0.0
        pursuing=sum(1 for u in sim.intercepts if u.status==1)
        vals=[str(sim.kills),str(sim.misses),str(sim.breaches),
              f"{sr:.1f}%",f"{med_lat:.1f}ms",f"{p99_lat:.1f}ms",
              f"{mean_rmse:.2f}m",str(len(active_t)),
              f"{pursuing}/{NUM_INTERCEPTORS}",
              str(sim.kill_streak),str(sim.best_streak),
              str(sim.wave),f"{sim.t:.0f}s"]
        for sv,v in zip(stat_vals,vals): sv.set_text(v)

        for i,u in enumerate(sim.intercepts):
            fuel_fills[i].set_width(u.fuel)
            fc=PAL["red"] if u.low_fuel else (PAL["amber"] if u.fuel<0.4 else u.color)
            fuel_fills[i].set_facecolor(fc)
            st=status_names[min(u.status,5)]
            sc=status_cols[min(u.status,5)]
            status_labels[i].set_text(st); status_labels[i].set_color(sc)
            wh="▲"*u.warheads+"△"*(6-u.warheads)
            warhead_txts[i].set_text(wh)
            warhead_txts[i].set_color(PAL["green"] if u.warheads>0 else PAL["red"])

        evlist=list(sim.events)
        for i,ll in enumerate(log_txts):
            if i<len(evlist):
                ev=evlist[i]
                col=(PAL["green"] if "KILL" in ev else
                     PAL["red"]   if "BREACH" in ev else
                     PAL["amber"] if "WAVE" in ev else PAL["cyan"])
                ll.set_text(ev[:48]); ll.set_color(col)
                ll.set_alpha(max(0.25,1.0-i*0.115))
            else: ll.set_text("")

        return []

    ani=animation.FuncAnimation(fig,update,frames=MAX_FRAMES,
                                  interval=ANIM_INTERVAL,blit=False,repeat=False)
    plt.subplots_adjust(left=0.01,right=0.99,top=0.93,bottom=0.03,wspace=0.22)
    return fig,ani


# ══════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    import sys
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    AEGIS-X v6.0  —  HIGH ACCURACY DRONE INTERCEPTION SYSTEM    ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  ACCURACY FIXES:                                                ║")
    print("║  • INTERCEPT_R  7.5m  → 18m   (+140%)                         ║")
    print("║  • FRAG_RADIUS  22m   → 40m   (+82%)                          ║")
    print("║  • FRAG_KILL_P  35%   → 85%   (+143%)                         ║")
    print("║  • PN gain      4.0   → 6.0   (+50%)                          ║")
    print("║  • APN factor   0.5   → 1.0   (+100%)                         ║")
    print("║  • Max speed    32    → 42 m/s(+31%)                          ║")
    print("║  • Max accel    22    → 35 m/s²(+59%)                         ║")
    print("║  • Warheads     3     → 6                                      ║")
    print("║  • Reload time  8/4   → 4/2 ticks                             ║")
    print("║  • Seeker FOV   50°   → 70°                                   ║")
    print("║  • Salvo-2 vs high-priority threats                            ║")
    print("║  • ECM 50% Pd reduction (not near-zero)                        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("  Building terrain DEM …",end="",flush=True); _=TERRAIN; print(" done")
    print("  Initialising atmosphere …",end="",flush=True); _=ATM; print(" done")
    print("  Starting simulation …\n")
    fig,ani=run()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
        sys.exit(0)