import React, { useState, useEffect } from 'react';

/* ========================================================================
   ICONS (SVG Manual)
   ======================================================================== */
const Icons = {
  Menu: () => (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
  ),
  Close: () => (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
  ),
  Search: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
  ),
  ArrowRight: () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
  ),
  Code: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
  ),
  Brain: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
  ),
  Download: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
  ),
  Message: () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
  )
};

/* ========================================================================
   COMPONENTS
   ======================================================================== */

/* --- Navbar --- */
const Navbar = ({ toggleTheme, isDark }) => {
  const [isOpen, setIsOpen] = useState(false);
  const links = ['Home', 'About', 'Experience', 'Projects', 'Contact'];

  return (
    <nav className="fixed top-6 left-0 right-0 z-50 flex justify-center px-4">
      {/* LIGHT: bg-white, border-2 black, shadow kasar 
         DARK: glassmorphism, border halus
      */}
      <div className="w-full max-w-7xl 
        bg-white dark:bg-[#0a0a0a]/90 
        border-2 border-black dark:border-white/10 
        shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-2xl 
        rounded-full px-6 py-3 flex justify-between items-center 
        transition-all duration-300 backdrop-blur-xl">
        
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-black dark:bg-white rounded-full flex items-center justify-center text-white dark:text-black border-2 border-transparent dark:border-transparent transition-colors">
            <span className="font-black text-lg">AI</span>
          </div>
          <span className="text-lg font-bold tracking-tighter text-black dark:text-white transition-colors">
            Engineer<span className="text-gray-500">.dev</span>
          </span>
        </div>

        {/* Desktop Menu */}
        <div className="hidden md:flex gap-1">
          {links.map((item) => (
            <a 
              key={item} 
              href={`#${item.toLowerCase()}`} 
              className="px-5 py-2 rounded-full text-xs font-bold 
              text-black dark:text-gray-400 
              hover:bg-black hover:text-white dark:hover:text-white dark:hover:bg-white/5 
              border border-transparent hover:border-black dark:hover:border-transparent
              transition-all"
            >
              {item}
            </a>
          ))}
        </div>

        {/* Action / Theme Toggle */}
        <div className="flex items-center gap-3">
          {/* THEME TOGGLE BUTTON (Comic Style vs Futuristic) */}
          <button 
            onClick={toggleTheme}
            className="w-10 h-10 flex items-center justify-center rounded-full 
            bg-white dark:bg-transparent
            border-2 border-black dark:border-white/10
            shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] dark:shadow-none
            hover:translate-y-[2px] hover:shadow-none hover:bg-yellow-200 dark:hover:bg-white/10
            transition-all text-lg"
            aria-label="Toggle Theme"
            title={isDark ? "Switch to Comic Mode" : "Switch to Dark Mode"}
          >
            {isDark ? "☀️" : "🖊️"}
          </button>
          
          <button className="hidden md:block 
            bg-black dark:bg-white 
            text-white dark:text-black 
            border-2 border-transparent 
            px-5 py-2 rounded-full text-xs font-bold 
            hover:opacity-80 transition-opacity shadow-lg">
            Hire Me
          </button>
          
          <button onClick={() => setIsOpen(!isOpen)} className="md:hidden text-black dark:text-white p-2">
            {isOpen ? <Icons.Menu /> : <Icons.Close />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="absolute top-24 left-4 right-4 
          bg-white dark:bg-[#141414] 
          border-2 border-black dark:border-white/10 
          shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-2xl
          rounded-3xl p-6 flex flex-col gap-2 md:hidden z-50">
           {links.map((item) => (
            <a key={item} href={`#${item.toLowerCase()}`} onClick={() => setIsOpen(false)} className="text-black dark:text-gray-300 hover:bg-black hover:text-white dark:hover:bg-white/10 px-4 py-3 rounded-xl font-bold transition-colors">
              {item}
            </a>
          ))}
        </div>
      )}
    </nav>
  );
};

/* --- Hero --- */
const Hero = () => {
  return (
    <section id="home" className="space-y-6 pt-6">
      {/* Top Widget Bar */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        <div className="md:col-span-8 
          bg-white dark:bg-[#141414] 
          rounded-full px-8 py-4 flex items-center gap-4 
          border-2 border-black dark:border-white/5 
          shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
          text-gray-500 transition-colors duration-300">
          <Icons.Search />
          <input type="text" placeholder="Search ML models, datasets, or papers..." className="bg-transparent border-none outline-none w-full text-black dark:text-white text-sm font-medium" />
        </div>
        <div className="md:col-span-4 
          bg-white dark:bg-[#141414] 
          rounded-full px-8 py-4 flex items-center justify-between 
          border-2 border-black dark:border-white/5 
          shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
          transition-colors duration-300">
          <span className="text-xs font-bold tracking-widest text-black dark:text-gray-500 uppercase">System Status</span>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse border border-black dark:border-transparent"></span>
            <span className="text-xs font-bold text-black dark:text-white">MODEL TRAINING</span>
          </div>
        </div>
      </div>

      {/* Main Hero Card */}
      <div className="
        bg-white dark:bg-[#141414] 
        rounded-[3rem] p-8 md:p-14 
        border-2 border-black dark:border-white/5 
        shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] dark:shadow-2xl 
        min-h-[500px] flex flex-col justify-between 
        relative overflow-hidden group transition-all duration-300">
        
        {/* Background Gradient (Visible only in Dark Mode) */}
        <div className="hidden dark:block absolute top-0 right-0 w-[400px] h-[400px] bg-blue-900/20 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/4 pointer-events-none"></div>

        <div className="grid md:grid-cols-2 gap-12 relative z-10 h-full">
          <div className="flex flex-col justify-center space-y-8">
            <div className="inline-flex items-center gap-2 
              bg-gray-100 dark:bg-white/5 
              border-2 border-black dark:border-white/10 
              w-max px-4 py-1.5 rounded-full transition-colors">
              <span className="text-xs font-bold text-black dark:text-white">v2.0 AI Portfolio</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-black text-black dark:text-white leading-[1.1] tracking-tight transition-colors">
              AI ENGINEER <br />
              <span className="text-black dark:text-transparent dark:bg-clip-text dark:bg-gradient-to-r dark:from-blue-400 dark:to-white decoration-4 underline decoration-black dark:decoration-transparent underline-offset-4">
                PORTFOLIO
              </span>
            </h1>

            <p className="text-black dark:text-gray-400 text-lg max-w-md transition-colors font-medium">
              Building intelligent systems with data, code, and design. Specialized in NLP, Computer Vision, and Predictive Analytics.
            </p>

            <div className="flex gap-4 pt-4">
              <button className="
                bg-black dark:bg-white 
                text-white dark:text-black 
                px-8 py-3 rounded-full text-sm font-bold 
                border-2 border-black dark:border-transparent
                hover:-translate-y-1 hover:shadow-[4px_4px_0px_0px_rgba(100,100,100,1)] dark:hover:shadow-none dark:hover:opacity-80 
                transition-all">
                View Projects
              </button>
              <button className="
                bg-white dark:bg-[#222] 
                text-black dark:text-white 
                px-8 py-3 rounded-full text-sm font-bold 
                border-2 border-black dark:border-white/10 
                hover:bg-gray-100 dark:hover:bg-[#333] 
                shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-none
                active:shadow-none active:translate-y-[4px]
                transition-all">
                Contact Me
              </button>
            </div>
          </div>

          {/* Code Visual */}
          <div className="flex flex-col justify-center">
             <div className="
               bg-white dark:bg-[#1a1a1a] 
               rounded-2xl border-2 border-black dark:border-white/10 
               p-6 font-mono text-sm 
               shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-2xl 
               transition-colors">
                <div className="flex gap-2 mb-4 border-b-2 border-black dark:border-transparent pb-4">
                   <div className="w-3 h-3 rounded-full bg-black dark:bg-red-500 border border-black dark:border-transparent"></div>
                   <div className="w-3 h-3 rounded-full bg-white dark:bg-yellow-500 border border-black dark:border-transparent"></div>
                   <div className="w-3 h-3 rounded-full bg-gray-300 dark:bg-green-500 border border-black dark:border-transparent"></div>
                </div>
                <div className="space-y-2 text-black dark:text-gray-400 font-bold dark:font-normal">
                   <p><span className="text-blue-600 dark:text-blue-400">import</span> torch</p>
                   <p><span className="text-blue-600 dark:text-blue-400">import</span> tensorflow <span className="text-blue-600 dark:text-blue-400">as</span> tf</p>
                   <p className="text-gray-500 italic"># Initializing Neural Network</p>
                   <p><span className="text-purple-600 dark:text-purple-400">class</span> <span className="text-yellow-600 dark:text-yellow-400">AI_Model</span>(nn.Module):</p>
                   <p className="pl-4">def __init__(self):</p>
                   <p className="pl-8">super(AI_Model, self).__init__()</p>
                   <p className="pl-8 text-green-600 dark:text-green-400">self.status = "Ready to Deploy"</p>
                </div>
             </div>
          </div>
        </div>
      </div>
    </section>
  );
};

/* --- About --- */
const About = () => {
  return (
    <section id="about" className="
      bg-white dark:bg-[#141414] 
      rounded-[2.5rem] p-10 
      border-2 border-black dark:border-white/5 
      shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
      relative overflow-hidden transition-colors duration-300">
      <div className="grid md:grid-cols-2 gap-10 items-center">
        <div>
          <h2 className="text-3xl font-bold text-black dark:text-white mb-6">About My Tech Stack</h2>
          <p className="text-black dark:text-gray-400 text-sm leading-relaxed mb-6 font-medium">
             I am an engineer focused on the intersection of Software Engineering and Artificial Intelligence. 
             My goal is to create scalable AI solutions that solve real-world problems using state-of-the-art algorithms.
          </p>
          <div className="grid grid-cols-2 gap-4">
             {['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'SQL', 'FastAPI'].map((tech) => (
                <div key={tech} className="flex items-center gap-2 text-black dark:text-gray-300 text-sm font-bold">
                   <div className="w-2 h-2 rounded-full bg-black dark:bg-blue-500"></div> {tech}
                </div>
             ))}
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
           {[
             { title: "Machine Learning", desc: "Supervised & Unsupervised" },
             { title: "Deep Learning", desc: "CNNs, RNNs, Transformers" },
             { title: "Data Engineering", desc: "ETL Pipelines & Big Data" },
             { title: "Model Deployment", desc: "Docker, Kubernetes, AWS" }
           ].map((item, idx) => (
              <div key={idx} className="
                bg-white dark:bg-[#0a0a0a] 
                p-5 rounded-2xl 
                border-2 border-black dark:border-white/5 
                shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
                hover:-translate-y-1 transition-all">
                 <div className="text-black dark:text-white"><Icons.Brain /></div>
                 <h3 className="text-black dark:text-white font-bold mt-3 mb-1">{item.title}</h3>
                 <p className="text-xs text-gray-600 dark:text-gray-500 font-semibold">{item.desc}</p>
              </div>
           ))}
        </div>
      </div>
    </section>
  );
};

/* --- Experience --- */
const Experience = () => {
  const steps = [
    { title: "Foundations", subtitle: "Math & Stats", desc: "Mastering Linear Algebra, Calculus, and Probability." },
    { title: "Data Analysis", subtitle: "Python & Pandas", desc: "Exploratory Data Analysis and visualization techniques." },
    { title: "Machine Learning", subtitle: "Algorithms", desc: "Implementing regression, classification, and clustering." },
    { title: "Deep Learning", subtitle: "Neural Networks", desc: "Building complex models for Vision and NLP tasks." }
  ];

  return (
    <section id="experience" className="space-y-6">
       <div className="
         bg-white dark:bg-[#141414] 
         rounded-[2.5rem] p-8 md:p-10 
         border-2 border-black dark:border-white/5 
         shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
         transition-colors duration-300">
          <div className="flex items-center gap-3 mb-8">
             <div className="bg-black dark:bg-white text-white dark:text-black px-3 py-1 rounded-full text-xs font-bold border-2 border-transparent">TIMELINE</div>
             <h2 className="text-2xl font-bold text-black dark:text-white">Engineering Journey</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
             {steps.map((step, index) => (
                <div key={index} className="
                  bg-white dark:bg-[#0a0a0a] 
                  p-6 rounded-3xl 
                  border-2 border-black dark:border-white/5 
                  shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
                  hover:-translate-y-1 transition-transform">
                   <div className="
                     w-10 h-10 rounded-full 
                     bg-black dark:bg-[#1a1a1a] 
                     border border-black dark:border-white/10 
                     flex items-center justify-center 
                     text-white font-bold mb-4">
                      {index + 1}
                   </div>
                   <h3 className="text-lg font-bold text-black dark:text-white">{step.title}</h3>
                   <span className="text-xs font-mono text-blue-600 dark:text-blue-400 block mb-2 font-bold">{step.subtitle}</span>
                   <p className="text-sm text-black dark:text-gray-500 leading-snug font-medium">{step.desc}</p>
                </div>
             ))}
          </div>
       </div>
    </section>
  );
};

/* --- Projects --- */
const Projects = () => {
  const projects = [
    { title: "Predictive Analytics", tech: "Python, Scikit-learn", desc: "Forecasting market trends using historical data." },
    { title: "Computer Vision", tech: "OpenCV, PyTorch", desc: "Real-time object detection system for security feeds." },
    { title: "NLP Chatbot", tech: "LLM, Transformers", desc: "Context-aware customer support bot using RAG." }
  ];

  return (
    <section id="projects" className="
      bg-white dark:bg-[#141414] 
      rounded-[2.5rem] p-8 md:p-12 
      border-2 border-black dark:border-white/5 
      shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
      transition-colors duration-300">
       <div className="flex justify-between items-end mb-8">
          <div>
             <span className="text-xs font-bold text-black dark:text-gray-500 uppercase tracking-widest">Selected Works</span>
             <h2 className="text-3xl font-bold text-black dark:text-white mt-2">Projects</h2>
          </div>
          <button className="hidden md:flex items-center gap-2 text-sm font-bold text-black dark:text-white hover:underline decoration-2">
             View GitHub <Icons.ArrowRight />
          </button>
       </div>

       <div className="grid md:grid-cols-3 gap-6">
          {projects.map((proj, idx) => (
             <div key={idx} className="group 
               bg-white dark:bg-[#0a0a0a] 
               rounded-[2rem] 
               border-2 border-black dark:border-white/5 
               shadow-[5px_5px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
               p-1 relative overflow-hidden 
               hover:translate-x-1 hover:translate-y-1 hover:shadow-none dark:hover:border-white/20 dark:hover:shadow-none
               transition-all">
                <div className="bg-white dark:bg-[#111] rounded-[1.8rem] p-6 h-full flex flex-col justify-between transition-colors">
                   <div className="mb-8">
                      <div className="w-10 h-10 bg-gray-100 dark:bg-[#222] rounded-full flex items-center justify-center mb-4 text-black dark:text-white border border-black dark:border-transparent transition-colors">
                         <Icons.Code />
                      </div>
                      <h3 className="text-xl font-bold text-black dark:text-white mb-2">{proj.title}</h3>
                      <p className="text-black dark:text-gray-400 text-sm leading-relaxed font-medium">{proj.desc}</p>
                   </div>
                   <div className="pt-4 border-t-2 border-black dark:border-white/5 flex justify-between items-center">
                      <span className="text-xs font-mono text-gray-600 dark:text-gray-500 font-bold">{proj.tech}</span>
                      <div className="w-8 h-8 rounded-full bg-black dark:bg-white text-white dark:text-black flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                         <Icons.ArrowRight />
                      </div>
                   </div>
                </div>
             </div>
          ))}
       </div>
    </section>
  );
};

/* --- Contact --- */
const Contact = () => {
  return (
    <section id="contact" className="
      bg-white dark:bg-[#141414] 
      rounded-[2.5rem] p-8 md:p-12 
      border-2 border-black dark:border-white/5 
      shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
      transition-colors duration-300 flex flex-col md:flex-row items-center justify-between gap-8">
      
      <div className="space-y-4">
        <h2 className="text-3xl font-bold text-black dark:text-white">Let's Work Together</h2>
        <p className="text-black dark:text-gray-400 text-sm max-w-md font-medium">
          Interested in building something intelligent? I'm currently open for new opportunities and collaborations.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 w-full md:w-auto">
        <button className="
          flex items-center justify-center gap-2
          bg-black dark:bg-white 
          text-white dark:text-black 
          px-8 py-4 rounded-xl text-sm font-bold 
          border-2 border-black dark:border-transparent
          hover:-translate-y-1 hover:shadow-[4px_4px_0px_0px_rgba(100,100,100,1)] dark:hover:shadow-none dark:hover:opacity-80 
          transition-all">
          <Icons.Download /> Download CV
        </button>
      </div>
    </section>
  );
};

/* --- Footer --- */
const Footer = () => {
  return (
    <footer className="
      bg-white dark:bg-[#141414] 
      rounded-full px-8 py-6 
      border-2 border-black dark:border-white/5 
      shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] dark:shadow-none 
      flex flex-col md:flex-row items-center justify-between gap-4 
      transition-colors duration-300">
       <div className="flex items-center gap-2 text-xs font-bold text-black dark:text-gray-500 uppercase">
          <div className="w-6 h-6 rounded-full bg-black dark:bg-white flex items-center justify-center text-white dark:text-black font-black text-[10px]">AI</div>
          <span>&copy; 2025 AI Engineer.</span>
       </div>
       <div className="flex gap-6 text-xs font-bold text-black dark:text-gray-500">
          <a href="#" className="hover:underline decoration-2">GitHub</a>
          <a href="#" className="hover:underline decoration-2">LinkedIn</a>
          <a href="#" className="hover:underline decoration-2">Email</a>
       </div>
    </footer>
  );
};

/* ========================================================================
   MAIN APP
   ======================================================================== */
const App = () => {
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') || 'dark';
    }
    return 'dark';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  useEffect(() => {
    document.documentElement.style.scrollBehavior = 'smooth';
  }, []);

  return (
    // Background: Comic Pattern for Light, Dark Hex for Dark
    <div className="min-h-screen bg-comic dark:bg-[#050505] text-black dark:text-white font-sans selection:bg-black selection:text-white dark:selection:bg-blue-500 pb-4 overflow-x-hidden transition-colors duration-300">
      
      <Navbar toggleTheme={toggleTheme} isDark={theme === 'dark'} />
      
      <main className="relative z-10 max-w-7xl mx-auto px-4 pt-28 space-y-6">
         <Hero />
         <About />
         <Experience />
         <Projects />
         <Contact />
         <Footer />
      </main>
    </div>
  );
};

export default App;