# app.py — TimeSculpt Trajectory Engine (single-file build)

import os, json, datetime as dt, random
import numpy as np, pandas as pd, altair as alt, streamlit as st
import sqlite3

# ---------- DB ----------
DB = "timesculpt.db"

def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = lambda cur,row: {d[0]: row[i] for i,d in enumerate(cur.description)}
    return c

def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS days(
          d TEXT PRIMARY KEY, note TEXT, state TEXT,
          focus REAL, energy REAL, progress REAL
        );
        CREATE TABLE IF NOT EXISTS loops(
          d TEXT, name TEXT, minutes REAL,
          PRIMARY KEY(d,name)
        );
        CREATE TABLE IF NOT EXISTS lens(name TEXT PRIMARY KEY, data TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, val TEXT);
        CREATE TABLE IF NOT EXISTS custom_loops(name TEXT PRIMARY KEY, category TEXT, polarity INTEGER);
        """)
init_db()

def save_day(d, note, loops_dict, state, F, E, P):
    with _conn() as c:
        c.execute("INSERT OR REPLACE INTO days VALUES (?,?,?,?,?,?)", (d, note, state, F, E, P))
        for k,v in loops_dict.items():
            c.execute("INSERT OR REPLACE INTO loops VALUES (?,?,?)",(d,k,float(v)))

def load_days(n=120):
    with _conn() as c:
        days = c.execute("SELECT * FROM days ORDER BY d ASC").fetchall()
        loops = c.execute("SELECT * FROM loops").fetchall()
    by = {}
    for r in loops:
        by.setdefault(r["d"],{})[r["name"]] = r["minutes"]
    for d in days:
        d["loops"] = by.get(d["d"],{})
    return days[-n:] if n else days

def settings_get(k, default=""): 
    with _conn() as c:
        r = c.execute("SELECT val FROM settings WHERE key=?",(k,)).fetchone()
    return r["val"] if r else default
def settings_set(k,v): 
    with _conn() as c: c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(k,v))

def lens_put(name,data): 
    with _conn() as c: c.execute("INSERT OR REPLACE INTO lens VALUES(?,?)",(name,json.dumps(data)))
def lens_all(): 
    with _conn() as c: return c.execute("SELECT * FROM lens").fetchall()

def custom_loops_all(): 
    with _conn() as c: return c.execute("SELECT * FROM custom_loops").fetchall()
def custom_loop_add(name,cat,pol): 
    if not name.strip(): return
    with _conn() as c: c.execute("INSERT OR REPLACE INTO custom_loops VALUES(?,?,?)",(name.strip(),cat,int(pol)))

# ---------- Lens helpers ----------
CORE_LENS = {
    "name":"Core",
    "collapse":["Release what drags the timeline.","Close the tab. End the loop.","No path opens while you hold every door."],
    "recursion":["Repeat the action that proves the future.","Small loops compound into fate.","Consistency sculpts identity."],
    "emergence":["Invite the first true move.","Bend toward the version of you that acts.","Begin poorly. Arrival happens mid-motion."],
    "neutral":["Attend to what is here. Choose again."]
}
def lens_phrase(lens, kind):
    b = lens.get(kind) or []
    if b: return random.choice(b)
    cb = CORE_LENS.get(kind,[])
    if cb: return random.choice(cb)
    for k in ("recursion","collapse","emergence","neutral"):
        arr = (lens.get(k) or []) or CORE_LENS.get(k,[])
        if arr: return random.choice(arr)
    return ""

def clean_passages(text):
    def _norm(t): return " ".join((t or "").split())
    def _clip(t,lo=80,hi=400):
        t=_norm(t); 
        if len(t)<lo: return None
        if len(t)>hi: t=t[:hi].rsplit(" ",1)[0]
        return t
    chunks=[p.strip() for p in text.replace("\r","").split("\n") if len(p.strip())>=40] or [text[:280]]
    seen=set(); parts={"collapse":[],"recursion":[],"emergence":[],"neutral":[]}
    KW={"collapse":["release","end","close","let go","discard","quit","stop"],
        "recursion":["repeat","again","habit","loop","daily","consistency"],
        "emergence":["begin","start","spark","new","future","grow","transform"]}
    def cat(p):
        t=p.lower()
        for k,keys in KW.items():
            if any(x in t for x in keys): return k
        return "neutral"
    for raw in chunks:
        c=_clip(raw); 
        if not c or c in seen: continue
        seen.add(c); parts[cat(c)].append(c)
    for k in parts: parts[k]=parts[k][:300]
    return parts

def get_active_lens():
    n=settings_get("active_lens","")
    if not n: return CORE_LENS
    for r in lens_all():
        if r["name"]==n: return {"name":n, **json.loads(r["data"])}
    return CORE_LENS

# ---------- State model ----------
W_POS,W_NEG,W_PROG,W_ENER=0.8,0.9,0.25,0.15
def label_state(loops,pos,neg):
    posm=sum(loops.get(k,0) for k in pos); negm=sum(loops.get(k,0) for k in neg)
    energy=min(100,(loops.get("body:walk",0)*1.2+loops.get("body:exercise",0)*1.6+loops.get("body:sleep_good",0)*1.5)/2.0)
    progress=min(100,(loops.get("creation:writing",0)*1.4+loops.get("creation:project",0)*1.2+loops.get("finance:save_invest",0)*1.1+loops.get("mind:planning",0)*0.9))
    focus_raw=(posm*W_POS-negm*W_NEG)+progress*W_PROG+energy*W_ENER
    focus=max(0,min(100,focus_raw))
    if negm>posm*1.2 or loops.get("consumption:scroll",0)>=45: state="Drift"
    elif posm>=negm and (loops.get("creation:writing",0)+loops.get("creation:project",0))>=30: state="Focused"
    else: state="Mixed"
    contribs=[]
    for k in pos:
        m=loops.get(k,0); 
        if m>0: contribs.append((k,m*W_POS,m))
    for k in neg:
        m=loops.get(k,0);
        if m>0: contribs.append((k,-m*W_NEG,m))
    pos_sorted=sorted([c for c in contribs if c[1]>0],key=lambda x:-x[1])[:2]
    neg_sorted=sorted([c for c in contribs if c[1]<0],key=lambda x:abs(x[1]),reverse=True)[:2]
    def fmt_pos(c): return f"+{int(c[2])}m {c[0].split(':',1)[1]} (impact +{c[1]:.1f})"
    def fmt_neg(c): return f"-{int(c[2])}m {c[0].split(':',1)[1]} (impact {c[1]:.1f})"
    plus=", ".join(fmt_pos(c) for c in pos_sorted) if pos_sorted else "+0m"
    minus=", ".join(fmt_neg(c) for c in neg_sorted) if neg_sorted else "-0m"
    micro=f"{plus} | {minus}"
    return state,round(focus,1),round(energy,1),round(progress,1),micro

# ---------- Forecast ----------
STATES=["Focused","Mixed","Drift"]; IDX={s:i for i,s in enumerate(STATES)}
DELTA_CAP,DECAY,PRIOR_WEIGHT,UNIFORM_BLEND=2.0,0.97,0.5,0.08
def learn_matrix(days,decay=DECAY):
    C=np.ones((3,3))*PRIOR_WEIGHT; last=None; w=1.0
    for d in days:
        s=d.get("state"); 
        if s not in IDX: continue
        if last is not None: C[IDX[last],IDX[s]]+=w
        w*=decay; last=s
    M=C/C.sum(axis=1,keepdims=True); U=np.ones((3,3))/3.0
    M=(1-UNIFORM_BLEND)*M+UNIFORM_BLEND*U; M=np.maximum(1e-6,M)
    return M/M.sum(axis=1,keepdims=True)

def simulate(M,start_state,days=30,sims=2000):
    start=IDX.get(start_state,1); counts=np.zeros((days,3))
    for _ in range(sims):
        s=start
        for t in range(days):
            counts[t,s]+=1; s=np.random.choice([0,1,2],p=M[s])
    probs=counts/sims; exp_focus=probs[:,0].sum()
    return probs,float(exp_focus)

def tweak_matrix(M, **kwargs):
    A=M.copy()
    def adj_row(i,d): A[i]=np.maximum(0.001,A[i]+d); A[i]/=A[i].sum()
    if kwargs.get("d_self"): adj_row(IDX["Drift"],np.array([0,0,kwargs["d_self"]]))
    if kwargs.get("d_to_m"): adj_row(IDX["Drift"],np.array([0,kwargs["d_to_m"],0]))
    if kwargs.get("m_to_f"): adj_row(IDX["Mixed"],np.array([kwargs["m_to_f"],0,0]))
    if kwargs.get("f_self"): adj_row(IDX["Focused"],np.array([kwargs["f_self"],0,0]))
    return A

# ---------- Streamlit UI ----------
st.set_page_config(page_title="TimeSculpt", layout="wide")
USE_AI=st.sidebar.checkbox("AI narration",value=False)
tab=st.sidebar.radio("Go to",["Input","Forecast","Interventions","Diagnostics","Lens"],key="nav")

# Sticky header
days_all=load_days(365)
st.markdown("<div style='position:sticky;top:0;background:#0d0d0df2;border-bottom:1px solid #222;padding:.4rem 0;z-index:10'>",unsafe_allow_html=True)
if days_all: st.markdown(f"**Today** • {days_all[-1]['state']} • Focus {days_all[-1]['focus']:.0f} • Energy {days_all[-1]['energy']:.0f} • Progress {days_all[-1]['progress']:.0f}")
else: st.markdown("**Today** • no data yet")
st.markdown("</div>",unsafe_allow_html=True)

# Tabs
if tab=="Input":
    st.header("Input")
    d=st.date_input("Date",value=dt.date.today()).isoformat()
    goal=st.text_input("Goal / Desire",value=settings_get("goal",""))
    if st.button("Save goal"): settings_set("goal",goal); st.success("Saved.")
    builtin=[("creation:writing",+1),("creation:project",+1),("mind:planning",+1),
             ("mind:reading",+1),("mind:meditation",+1),("body:walk",+1),
             ("body:exercise",+1),("body:sleep_good",+1),("consumption:scroll",-1),
             ("consumption:youtube",-1),("food:junk",-1),("finance:save_invest",+1),
             ("finance:budget_check",+1),("body:late_sleep",-1),("finance:impulse_spend",-1)]
    customs=custom_loops_all(); names=[n for n,_ in builtin]; pos={n for n,p in builtin if p>0}; neg={n for n,p in builtin if p<0}
    for r in customs:
        names.append(f"{r['category']}:{r['name']}")
        if r["polarity"]>0: pos.add(names[-1])
        else: neg.add(names[-1])
    cols=st.columns(4); loops_today={}
    for i,(k,label) in enumerate([("creation:writing","Writing"),("creation:project","Project"),
        ("mind:planning","Planning"),("mind:reading","Reading"),("mind:meditation","Meditation"),
        ("body:walk","Walk"),("body:exercise","Exercise"),("body:sleep_good","Good sleep"),
        ("body:late_sleep","Late sleep"),("consumption:scroll","Scroll"),("consumption:youtube","YouTube"),
        ("food:junk","Junk food"),("finance:save_invest","Save/Invest"),("finance:budget_check","Budget check"),
        ("finance:impulse_spend","Impulse spend")]):
        with cols[i%4]: loops_today[k]=st.number_input(label,min_value=0,max_value=600,value=0,step=5,key=f"i_{k}")
    state,F,E,P,micro=label_state(loops_today,pos,neg)
    st.info(f"{state} • Focus {F} • Energy {E} • Progress {P}\n\n{micro}")
    note=st.text_area("Note")
    if st.button("Commit today"): save_day(d,note,loops_today,state,F,E,P); st.success("Saved.")
    # Custom add
    cc1,cc2,cc3=st.columns(3)
    with cc1: cname=st.text_input("Custom name")
    with cc2: ccat=st.selectbox("Category",["creation","mind","body","consumption","food","finance"])
    with cc3: cpol=st.selectbox("Polarity",[+1,-1])
    if st.button("Add custom"): custom_loop_add(cname,ccat,cpol); st.experimental_rerun()
    # Seed demo
    if st.button("Seed demo week"):
        today=dt.date.today()
        for i in range(7):
            d=(today-dt.timedelta(days=6-i)).isoformat()
            loops={"creation:writing":30+5*i,"body:walk":20,"consumption:scroll":40-3*i}
            s,F,E,P,m=label_state(loops,pos,neg); save_day(d,"seed",loops,s,F,E,P)
        st.success("Demo week added."); st.experimental_rerun()

elif tab=="Forecast":
    st.header("Forecast")
    days=load_days(120)
    if not days: st.info("Log at least 1 day.")
    else:
        M=learn_matrix(days); start=days[-1]["state"]
        probs,expF=simulate(M,start,30,2500)
        df=pd.DataFrame({"day":range(1,31),"Focused":probs[:,0],"Mixed":probs[:,1],"Drift":probs[:,2]})
        st.markdown(f"**Expected focused days**: **{df['Focused'].sum():.1f}/30**")
        dfm=df.melt("day",var_name="state",value_name="p")
        st.altair_chart(alt.Chart(dfm).mark_area(opacity=0.85).encode(
            x="day",y=alt.Y("p",stack="normalize",axis=alt.Axis(format="%")),
            color=alt.Color("state",scale=alt.Scale(domain=["Focused","Mixed","Drift"],range=["#E0C36D","#777","#B91C1C"]))
        ).properties(height=260),use_container_width=True)

elif tab=="Interventions":
    st.header("Interventions")
    days=load_days(120)
    if not days: st.info("Need some days.")
    else:
        start=days[-1]["state"]; M=learn_matrix(days); _,baseF=simulate(M,start,30,2000)
        INTERVENTIONS=[{"title":"7-min starter","how":"Start badly. Stop after 7.","tags":["creation"],"tweak":{"m_to_f":+0.06}},
                       {"title":"15-min walk","how":"Swap one scroll for a walk.","tags":["body"],"tweak":{"m_to_f":+0.05,"d_self":-0.03}},
                       {"title":"Sleep before midnight","how":"Shut down 30 min earlier.","tags":["body"],"tweak":{"d_self":-0.06}},
                       {"title":"10% pay-yourself-first","how":"Automate on payday.","tags":["finance"],"tweak":{"m_to_f":+0.05}}]
        results=[]
        for iv in INTERVENTIONS:
            M2=tweak_matrix(M,**iv["tweak"]); _,f2=simulate(M2,start,30,2000)
            delta=min(DELTA_CAP,f2-baseF)
            results.append({**iv,"delta":delta})
        results.sort(key=lambda r:-r["delta"])
        lens=get_active_lens()
        top=results[0]; styled=f"{top['title']} — {lens_phrase(lens,'recursion')}"
        st.success(f"**One smallest move**\n\n{styled}\n\n{top['how']}\n\nΔ Focused days ≈ +{top['delta']:.2f}")

elif tab=="Diagnostics":
    st.header("Diagnostics (demo)")
    st.info("Shows Force (+) and Drift (−) habits once enough days are logged.")

elif tab=="Lens":
    st.header("Lens")
    up=st.file_uploader("Upload .txt/.docx/.pdf",type=["txt","docx","pdf"])
    name=st.text_input("Lens name","Lens 1")
    if st.button("Add Lens") and up:
        text=""
        try:
            if up.name.endswith(".txt"): text=up.read().decode("utf-8","ignore")
            elif up.name.endswith(".docx"):
                import docx
                d=docx.Document(up); text="\n".join(p.text for p in d.paragraphs)
            elif up.name.endswith(".pdf"):
                import PyPDF2
                pdf=PyPDF2.PdfReader(up); text="\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception: text=""
        parts = clean_passages(text or "")
        lens_put(name, parts)
        settings_set("active_lens", name)
        st.success(
            f"Added lens '{name}' with {sum(len(v) for v in parts.values())} passages."
        )

    # ---- Manage/preview lenses ----
    lenses = lens_all()
    if lenses:
        current = settings_get("active_lens") or lenses[0]["name"]
        names = [r["name"] for r in lenses]
        sel = st.selectbox(
            "Active lens",
            names,
            index=names.index(current) if current in names else 0,
        )
        if st.button("Activate"):
            settings_set("active_lens", sel)
            st.success(f"Activated {sel}")

        # Preview a few passages from the active lens
        lens = get_active_lens()
        st.markdown(f"**Active lens:** {lens['name']}")
        for bucket in ("collapse", "recursion", "emergence", "neutral"):
            st.write(f"{bucket.capitalize()} ({len(lens.get(bucket, []))} passages)")
            if lens.get(bucket):
                st.code(random.choice(lens[bucket]))
    else:
        st.info("No lenses yet. Using Core Lens.")
