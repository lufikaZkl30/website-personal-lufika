import React, { useState, useEffect } from 'react';

/**
 * =========================================================================
 * 🎨 1. GLOBAL CSS (src/index.css)
 * =========================================================================
 * Instruksi:
 * Copy isi string di dalam tag <style> ini ke file 'src/index.css'
 * jika kamu menjalankannya di local project.
 */
const Styles = () => (
  <style>{`
    /* Tailwind directives (optional - environment ini menanganinya otomatis) */
    /* @tailwind base; @tailwind components; @tailwind utilities; */

    :root {
      --bg: #f3f3f3;
      --black-ui: #0b0b0b;
    }

    body {
      margin: 0;
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: var(--bg);
      /* Center layout for preview */
      display: flex;
      justify-content: center;
      min-height: 100vh;
    }

    /* Container */
    .container-custom {
      max-width: 1100px;
      width: 100%;
      margin: 40px auto;
      padding: 20px;
    }

    /* --- Reusable Styles (Glossy UI) --- */
    
    /* Navbar Pill */
    .nav-pill {
      background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.04));
      box-shadow: 0 8px 20px rgba(0,0,0,0.15);
      border-radius: 9999px;
      padding: 10px 18px;
      display: inline-flex;
      gap: 12px;
      align-items: center;
      backdrop-filter: blur(10px);
    }

    .nav-link {
      color: #555;
      cursor: pointer;
      font-weight: 500;
      transition: color 0.2s;
    }
    .nav-link:hover { color: #000; }
    .nav-link.active {
      color: #000;
      font-weight: 700;
      text-decoration: underline;
    }

    /* Search Pill */
    .search-pill {
      background: linear-gradient(90deg, #0a0a0a, #111);
      border-radius: 40px;
      padding: 12px 18px;
      color: #ddd;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 8px 20px rgba(0,0,0,0.35), inset 0 4px 12px rgba(255,255,255,0.02);
    }

    /* Hero Card */
    .hero-card {
      background: linear-gradient(90deg, #0b0b0b, #111);
      border-radius: 28px;
      padding: 28px;
      color: white;
      box-shadow: 0 20px 40px rgba(0,0,0,0.45);
      overflow: hidden;
      position: relative;
    }

    .hero-title {
      font-size: 56px;
      font-weight: 800;
      letter-spacing: 2px;
      line-height: 1.05;
    }

    .badge-round {
      background: white;
      color: black;
      width: 44px;
      height: 44px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 9999px;
      font-weight: 700;
      box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }

    /* Info Card */
    .info-card {
      background: #fff;
      color: #111;
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 18px 30px rgba(0,0,0,0.25);
    }

    /* Icons */
    .icon-circle {
      background: #fff;
      width: 56px;
      height: 56px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 10px 20px rgba(0,0,0,0.25);
      margin-right: 12px;
      cursor: pointer;
      transition: transform 0.2s;
      color: #111;
      font-size: 20px;
    }
    .icon-circle:hover { transform: scale(1.05); }

    /* Timeline */
    .timeline-card {
      background: linear-gradient(90deg, #0b0b0b, #111);
      border-radius: 24px;
      padding: 28px;
      color: white;
      box-shadow: 0 18px 30px rgba(0,0,0,0.32);
    }

    .timeline-pill {
      background: #fff;
      color: #111;
      border-radius: 9999px;
      padding: 10px 18px;
      display: inline-block;
      margin: 6px;
      font-weight: 600;
      font-size: 14px;
    }

    /* Carousel */
    .carousel-card {
      background: linear-gradient(145deg, #1a1a1a, #000);
      border-radius: 18px;
      padding: 16px;
      min-width: 220px;
      height: 130px;
      box-shadow: 0 18px 30px rgba(0,0,0,0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
    }

    /* Footer */
    .footer-pill {
      background: linear-gradient(90deg, #0b0b0b, #111);
      padding: 14px 30px;
      border-radius: 9999px;
      color: white;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: 0 20px 40px rgba(0,0,0,0.35);
      width: 100%;
      position: relative;
    }

    /* Buttons */
    .btn-pill {
      background: #fff;
      color: #111;
      padding: 10px 24px;
      border-radius: 28px;
      box-shadow: 0 10px 20px rgba(0,0,0,0.15);
      font-weight: 700;
      border: none;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn-pill:hover { background: #eee; }

    .btn-dark {
      background: #111;
      color: white;
      padding: 10px 24px;
      border-radius: 28px;
      box-shadow: 0 10px 20px rgba(0,0,0,0.2);
      border: none;
      cursor: pointer;
    }

    /* Responsive */
    @media (max-width: 768px) {
      .hero-title { font-size: 34px; }
      .container-custom { padding: 12px; margin-top: 10px; }
      .nav-pill { overflow-x: auto; max-width: 100%; }
      .hero-card > div { flex-direction: column; }
      .hero-card div[style*="width:360"] { width: 100% !important; }
    }
  `}</style>
);


/**
 * =========================================================================
 * 🧩 2. COMPONENTS
 * =========================================================================
 */

// --- src/components/Navbar.js ---
const Navbar = () => {
  return (
    <div style={{display:'flex', flexDirection: 'row', flexWrap: 'wrap', justifyContent:'space-between', alignItems:'center', gap:16}}>
      <div className="nav-pill">
        <div className="nav-link">Menu</div>
        <div className="nav-link active">Home</div>
        <div className="nav-link">Projects</div>
        <div className="nav-link">Gallery</div>
        <div className="nav-link">About</div>
      </div>
      <div style={{display:'flex', alignItems:'center', gap:12}}>
        <div className="nav-pill" style={{padding:'6px 12px'}}>
          <div className="nav-link" style={{color:'#000', margin:0}}>Account</div>
        </div>
        <div className="nav-pill search-pill" style={{padding:'6px'}}>
          <div style={{display:'flex', alignItems:'center', gap:8}}>
            <div style={{width:32,height:32,borderRadius:9999,background:'#fff',display:'flex',alignItems:'center',justifyContent:'center',color:'#000',fontWeight:700}}>B</div>
            <div style={{color:'#fff',fontWeight:600, paddingRight: 8}}>BellArt__03</div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- src/components/SearchBar.js ---
const SearchBar = () => {
  return (
    <div style={{display:'flex', flexWrap:'wrap', alignItems:'center', justifyContent:'space-between', gap:16}}>
      <div className="search-pill" style={{flex:1, minWidth: '280px'}}>
        <input className="" placeholder="Search history, articles..." style={{background:'transparent',border:'none',outline:'none',color:'#ddd',flex:1, fontSize: '14px'}} />
        <div style={{display:'flex',gap:12,alignItems:'center', paddingLeft: 10, borderLeft: '1px solid #333'}}>
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ddd" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ddd" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
        </div>
      </div>
      <div style={{color:'#888', fontWeight:700, letterSpacing:'0.2em', fontSize:'12px'}}>COLLABORATION</div>
    </div>
  );
};

// --- src/components/InfoCard.js ---
const InfoCard = () => {
  return (
    <div className="info-card">
      <div style={{display:'flex',justifyContent:'flex-end',gap:8,marginBottom:12}}>
        <button className="btn-dark">LOGIN</button>
        <button className="btn-pill" style={{background: '#f0f0f0'}}>GET STARTED</button>
      </div>
      <h3 style={{margin:'0 0 8px 0',fontSize:18,fontWeight:800}}>Apa itu UI/UX Design?</h3>
      <p style={{color:'#555', fontSize: '14px', lineHeight: '1.5'}}>Sejarah desain UI/UX dimulai dengan perkembangan teknologi dan kebutuhan akan antarmuka yang ramah pengguna.</p>
      <div style={{textAlign:'right',marginTop:12}}>
        <button style={{border:'none',background:'transparent',fontWeight:700,borderBottom:'2px solid #000',paddingBottom:2,cursor:'pointer', fontSize:'13px'}}>View more →</button>
      </div>
    </div>
  );
};

// --- src/components/HeroSection.js ---
const HeroSection = () => {
  return (
    <div className="hero-card" style={{display:'flex',gap:24,flexDirection:'column'}}>
      <div style={{display:'flex',gap:24,alignItems:'center', flexWrap: 'wrap'}}>
        <div style={{display:'flex',flexDirection:'column',flex:1, minWidth: '300px'}}>
          <div style={{display:'flex',alignItems:'center',gap:12,marginBottom:16}}>
            <div className="badge-round">0.1</div>
            <div style={{color:'#cfcfcf',fontWeight:700,letterSpacing:'0.08em', fontSize: '12px'}}>MENGENAL SEJARAH UI/UX</div>
          </div>
          <div className="hero-title">
            UI/UX DESIGN<br/>
            <span style={{background:'#fff',color:'#000',padding:'6px 10px',borderRadius:8,fontWeight:800,fontSize:14, verticalAlign: 'middle', marginLeft: '10px'}}>SEJARAH</span>
          </div>
        </div>
        <div style={{width:360, maxWidth: '100%'}}>
          <InfoCard />
        </div>
      </div>
      
      <div style={{marginTop: 10, height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', width: '80%', position: 'relative'}}>
         <div style={{position:'absolute', right:'-10px', top:'-10px', width:'24px', height:'24px', background:'white', borderRadius:'50%', border:'4px solid #000'}}></div>
      </div>

      <div style={{display:'flex',alignItems:'center',gap:12, marginTop: 10}}>
        <div className="icon-circle">❤</div>
        <div className="icon-circle">🔖</div>
        <div className="icon-circle">⤴</div>
      </div>
    </div>
  );
};

// --- src/components/TimelineSection.js ---
const TimelineSection = () => {
  return (
    <div className="timeline-card">
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:20, flexWrap:'wrap', gap:10}}>
        <h3 style={{margin:0,fontSize:20,fontWeight:800}}>TIMELINE SEJARAH</h3>
        <div style={{display:'flex',gap:8, flexWrap:'wrap'}}>
          <div className="timeline-pill">1. 1950-an</div>
          <div className="timeline-pill">2. 1968</div>
          <div className="timeline-pill" style={{background: '#333', color: 'white'}}>3. 1980-an</div>
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit, minmax(250px, 1fr))',gap:16}}>
        <div style={{background:'rgba(255,255,255,0.05)',padding:16,borderRadius:16, border: '1px solid rgba(255,255,255,0.1)'}}> 
          <h4 style={{color:'#fff',marginTop:0, marginBottom: 8}}>Era Mainframe</h4>
          <p style={{color:'#bbb', fontSize: '13px', lineHeight: '1.5'}}>Komputer pertama dirancang untuk militer. Antarmuka sangat kompleks dan hanya untuk teknisi terlatih.</p>
        </div>
        <div style={{background:'rgba(255,255,255,0.05)',padding:16,borderRadius:16, border: '1px solid rgba(255,255,255,0.1)'}}> 
          <h4 style={{color:'#fff',marginTop:0, marginBottom: 8}}>Kelahiran Mouse</h4>
          <p style={{color:'#bbb', fontSize: '13px', lineHeight: '1.5'}}>Douglas Engelbart menciptakan mouse dan konsep GUI pertama, mengubah cara interaksi manusia-komputer.</p>
        </div>
        <div style={{background:'rgba(255,255,255,0.05)',padding:16,borderRadius:16, border: '1px solid rgba(255,255,255,0.1)'}}> 
          <h4 style={{color:'#fff',marginTop:0, marginBottom: 8}}>Revolusi PC</h4>
          <p style={{color:'#bbb', fontSize: '13px', lineHeight: '1.5'}}>Apple meluncurkan Macintosh. Komputer personal dengan GUI yang memikat pengguna umum.</p>
        </div>
      </div>

      <div style={{marginTop:16,display:'flex',gap:8, flexWrap:'wrap'}}>
        <div className="timeline-pill" style={{background:'transparent',border:'1px solid rgba(255,255,255,0.2)',color:'#ddd'}}>4. 1990-an</div>
        <div className="timeline-pill" style={{background:'transparent',border:'1px solid rgba(255,255,255,0.2)',color:'#ddd'}}>5. 2000-an</div>
        <div className="timeline-pill">6. Saat Ini</div>
      </div>
    </div>
  );
};

// --- src/components/CarouselSection.js ---
const CarouselSection = () => {
  const items = [
    { title: 'Xerox PARC', bg: 'linear-gradient(135deg, #3b82f6, #1e3a8a)' },
    { title: 'Macintosh', bg: 'linear-gradient(135deg, #8b5cf6, #4c1d95)' },
    { title: 'Material Design', bg: 'linear-gradient(135deg, #10b981, #064e3b)' },
    { title: 'Modern Glassmorphism', bg: 'linear-gradient(135deg, #f59e0b, #78350f)' }
  ];

  return (
    <div style={{display:'flex',alignItems:'center',gap:12}}>
      <div className="icon-circle" style={{width:44,height:44,fontSize:20, flexShrink: 0}}>‹</div>
      <div style={{display:'flex',gap:12,overflowX:'auto',padding:'8px 4px', scrollbarWidth: 'none', width: '100%'}}>
        {items.map((item,i)=> (
          <div key={i} className="carousel-card" style={{background: item.bg, flexShrink: 0}}>
            {item.title}
          </div>
        ))}
      </div>
      <div className="icon-circle" style={{width:44,height:44,fontSize:20, flexShrink: 0}}>›</div>
    </div>
  );
};

// --- src/components/ProjectsSection.js ---
const ProjectsSection = () => {
  const [projects, setProjects] = useState(() => {
    try {
      const raw = localStorage.getItem('projects');
      return raw ? JSON.parse(raw) : [
        { id: 1, title: 'E-Commerce App', desc: 'Redesign aplikasi belanja online dengan fokus konversi.' },
        { id: 2, title: 'Banking Dashboard', desc: 'Dashboard admin untuk monitoring transaksi realtime.' }
      ];
    } catch { return []; }
  });
  const [form, setForm] = useState({title:'',desc:''});

  useEffect(() => {
    localStorage.setItem('projects', JSON.stringify(projects));
  }, [projects]);

  function addProject(e){
    e.preventDefault();
    if(!form.title) return alert('Masukkan judul project');
    const newP = { id: Date.now(), title: form.title, desc: form.desc };
    setProjects([newP, ...projects]);
    setForm({title:'',desc:''});
  }

  return (
    <div>
      <div style={{marginBottom: 20}}>
        <h3 style={{fontSize:20,fontWeight:800, marginBottom: 5}}>Projects</h3>
        <p style={{color:'#666', fontSize: '14px', margin: 0}}>Tambahkan project untuk portofoliomu — data disimpan lokal (localStorage).</p>
      </div>

      <form onSubmit={addProject} style={{background:'white', padding: 20, borderRadius: 16, boxShadow: '0 5px 15px rgba(0,0,0,0.05)', marginBottom: 24}}>
        <div style={{display:'flex', gap:10, flexWrap:'wrap'}}>
          <input 
            value={form.title} 
            onChange={e=>setForm({...form,title:e.target.value})} 
            placeholder="Judul Project (cth: Redesign Gojek)" 
            style={{padding:12,borderRadius:8,border:'1px solid #eee',flex:1, minWidth: '200px', outline: 'none', background: '#fafafa'}} 
          />
          <input 
            value={form.desc} 
            onChange={e=>setForm({...form,desc:e.target.value})} 
            placeholder="Deskripsi singkat..." 
            style={{padding:12,borderRadius:8,border:'1px solid #eee',flex:2, minWidth: '200px', outline: 'none', background: '#fafafa'}} 
          />
          <button className="btn-dark" type="submit">Tambah +</button>
        </div>
      </form>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))',gap:16}}>
        {projects.length === 0 && <div style={{color:'#777', fontStyle: 'italic'}}>Belum ada project. Tambahkan project di form di atas.</div>}
        {projects.map(p=> (
          <div key={p.id} style={{background:'white',padding:20,borderRadius:16,boxShadow:'0 8px 18px rgba(0,0,0,0.06)', border: '1px solid #f0f0f0'}}>
            <div style={{height: 40, width: 40, background: '#f3f3f3', borderRadius: 8, marginBottom: 12, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>📂</div>
            <h4 style={{margin:'0 0 8px 0',fontWeight:800}}>{p.title}</h4>
            <p style={{color:'#666', fontSize: '14px', margin: 0, lineHeight: '1.4'}}>{p.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

// --- src/components/GallerySection.js ---
const GallerySection = () => {
  // Placeholder images for preview
  const imgs = [
    'https://images.unsplash.com/photo-1561070791-2526d30994b5?auto=format&fit=crop&w=400&q=80',
    'https://images.unsplash.com/photo-1545235617-9465d2a55698?auto=format&fit=crop&w=400&q=80',
    'https://images.unsplash.com/photo-1586717791821-3f44a5638d0f?auto=format&fit=crop&w=400&q=80'
  ];
  return (
    <div>
      <h3 style={{fontSize:20,fontWeight:800, marginBottom: 5}}>Gallery</h3>
      <p style={{color:'#666', fontSize: '14px', marginTop: 0}}>Kumpulan screenshot / ilustrasi project.</p>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))',gap:16,marginTop:16}}>
        {imgs.map((src,i)=> (
          <div key={i} style={{background:'#e5e5e5',borderRadius:16,overflow:'hidden',height:160,display:'flex',alignItems:'center',justifyContent:'center', position: 'relative'}}>
            <img src={src} alt={`Project ${i+1}`} style={{width:'100%',height:'100%',objectFit:'cover', transition: 'transform 0.3s'}} />
            <div style={{position: 'absolute', bottom: 0, left: 0, right: 0, padding: 10, background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)', color: 'white', fontSize: '12px', fontWeight: 600}}>
              Project Shot {i+1}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// --- src/components/AboutSection.js ---
const AboutSection = () => {
  return (
    <div style={{background: 'white', padding: 30, borderRadius: 24, boxShadow: '0 10px 30px rgba(0,0,0,0.05)'}}>
      <h3 style={{fontSize:20,fontWeight:800, marginTop: 0}}>About Me</h3>
      <div style={{display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap'}}>
        <div style={{width: 80, height: 80, borderRadius: '50%', background: '#ddd', flexShrink: 0, overflow: 'hidden'}}>
           <img src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=150&q=80" alt="Profile" style={{width: '100%', height: '100%', objectFit: 'cover'}} />
        </div>
        <div style={{flex: 1}}>
          <p style={{color:'#555', lineHeight: '1.6', marginTop: 0}}>
            Halo! Saya mahasiswa Teknik Informatika semester akhir. Saya sangat tertarik pada dunia <b>UI/UX Design</b> dan <b>Frontend Development</b>. 
            Portofolio ini dibuat menggunakan React JS untuk mendemonstrasikan kemampuan saya dalam menerjemahkan desain menjadi kode yang fungsional.
          </p>
          <div style={{display: 'flex', gap: 10, marginTop: 15}}>
            <span style={{background: '#f3f3f3', padding: '6px 12px', borderRadius: 20, fontSize: '12px', fontWeight: 600}}>React JS</span>
            <span style={{background: '#f3f3f3', padding: '6px 12px', borderRadius: 20, fontSize: '12px', fontWeight: 600}}>Figma</span>
            <span style={{background: '#f3f3f3', padding: '6px 12px', borderRadius: 20, fontSize: '12px', fontWeight: 600}}>CSS3</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- src/components/ContactSection.js ---
const ContactSection = () => {
  const [sent, setSent] = useState(false);
  const [form, setForm] = useState({name:'',email:'',message:''});
  
  function submit(e){ e.preventDefault(); setSent(true); setForm({name:'',email:'',message:''}); }
  
  return (
    <div>
      <h3 style={{fontSize:20,fontWeight:800, marginBottom: 15}}>Contact</h3>
      {!sent ? (
        <form onSubmit={submit} style={{display:'grid',gap:12,maxWidth:600}}>
          <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12}}>
            <input required value={form.name} onChange={e=>setForm({...form,name:e.target.value})} placeholder="Nama Lengkap" style={{padding:14,borderRadius:12,border:'1px solid #ddd', outline: 'none'}} />
            <input required type="email" value={form.email} onChange={e=>setForm({...form,email:e.target.value})} placeholder="Alamat Email" style={{padding:14,borderRadius:12,border:'1px solid #ddd', outline: 'none'}} />
          </div>
          <textarea required value={form.message} onChange={e=>setForm({...form,message:e.target.value})} placeholder="Tulis pesan Anda di sini..." style={{padding:14,borderRadius:12,border:'1px solid #ddd',minHeight:120, outline: 'none', fontFamily: 'inherit'}} />
          <div style={{display: 'flex', justifyContent: 'flex-end'}}>
             <button className="btn-dark" type="submit" style={{padding: '12px 30px'}}>Kirim Pesan</button>
          </div>
        </form>
      ) : (
        <div style={{background:'#dcfce7',padding:20,borderRadius:16, color: '#166534', border: '1px solid #bbf7d0', textAlign: 'center'}}>
          <h4 style={{margin: '0 0 5px 0'}}>Pesan Terkirim!</h4>
          <p style={{margin: 0, fontSize: '14px'}}>Terima kasih telah menghubungi saya. Saya akan membalas secepatnya.</p>
          <button onClick={() => setSent(false)} style={{marginTop: 15, background: 'transparent', border: 'none', textDecoration: 'underline', cursor: 'pointer', color: '#166534', fontWeight: 600}}>Kirim pesan lain</button>
        </div>
      )}
    </div>
  );
};

// --- src/components/FooterNav.js ---
const FooterNav = () => {
  return (
    <div style={{display:'flex',justifyContent:'center', marginTop: 'auto'}}>
      <div className="footer-pill" style={{width:'100%',maxWidth:1100}}>
        <div style={{width:'33%',textAlign:'center',color:'#888',fontWeight:700, fontSize: '10px', letterSpacing: '1px'}}>SPECIAL COLLABORATION</div>
        <div style={{width:'33%',display:'flex',justifyContent:'center',position:'relative'}}>
          <div style={{position:'absolute',top:-42,background:'#fff',width:64,height:64,borderRadius:9999,display:'flex',alignItems:'center',justifyContent:'center',boxShadow:'0 10px 25px rgba(0,0,0,0.3)', cursor: 'pointer', transition: 'transform 0.2s'}}>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3z"/></svg>
          </div>
        </div>
        <div style={{width:'33%',textAlign:'center',color:'#888',fontWeight:700, fontSize: '10px', letterSpacing: '1px'}}>SEJARAH UI/UX DESIGN</div>
      </div>
    </div>
  );
};

/**
 * =========================================================================
 * 🚀 3. MAIN APP COMPONENT (src/App.js)
 * =========================================================================
 */
function App(){
  return (
    <>
      <Styles />
      <div className="container-custom">
        <Navbar />
        <div style={{height:32}} />
        <SearchBar />
        <div style={{height:32}} />
        <HeroSection />
        <div style={{height:40}} />
        <TimelineSection />
        <div style={{height:40}} />
        <ProjectsSection />
        <div style={{height:40}} />
        <GallerySection />
        <div style={{height:40}} />
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 40}}>
           <AboutSection />
           <ContactSection />
        </div>
        <div style={{height:80}} />
        <FooterNav />
      </div>
    </>
  );
}

export default App;