import React, { useState } from 'react';
import { 
  Menu, 
  Home, 
  Image as ImageIcon, 
  Save, 
  Info, 
  User, 
  Search, 
  Mic, 
  Camera, 
  Heart, 
  Bookmark, 
  Share2, 
  ArrowRight, 
  ChevronLeft, 
  ChevronRight, 
  LayoutTemplate, 
  Cpu, 
  Palette, 
  Code, 
  Award, 
  Mail 
} from 'lucide-react';

// --- COMPONENTS ---

const Navbar = () => (
  <div className="w-full flex flex-col md:flex-row justify-between items-center gap-4 relative z-50">
    {/* Main Nav Pill */}
    <div className="bg-black text-white rounded-full px-8 py-3 flex items-center gap-8 shadow-xl">
      <button className="hover:text-gray-300 transition font-medium text-sm flex items-center gap-2">
        <Menu size={18} /> Menu
      </button>
      <button className="hover:text-gray-300 transition font-medium text-sm border-b-2 border-white pb-0.5">Home</button>
      <button className="hover:text-gray-300 transition font-medium text-sm">Image</button>
      <button className="hover:text-gray-300 transition font-medium text-sm">Save</button>
      <button className="hover:text-gray-300 transition font-medium text-sm">About</button>
      <button className="hover:text-gray-300 transition font-medium text-sm">Account</button>
    </div>

    {/* Profile Card */}
    <div className="bg-black text-white rounded-full pl-2 pr-6 py-2 flex items-center gap-3 shadow-xl absolute right-0 md:relative">
      <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center text-black">
        <User size={18} />
      </div>
      <span className="font-bold text-sm">Luvi_Portfolio</span>
    </div>
  </div>
);

const SearchSection = () => (
  <div className="flex flex-col md:flex-row justify-between items-end md:items-center mt-8 mb-4 relative">
    {/* Search Bar */}
    <div className="bg-black text-gray-400 rounded-full px-6 py-3 w-full md:w-1/3 flex items-center justify-between shadow-lg">
      <span className="text-sm">Search..</span>
      <div className="flex gap-3 text-white">
        <Mic size={18} />
        <Camera size={18} />
      </div>
    </div>

    {/* Collaboration Label */}
    <div className="mt-4 md:mt-0 flex flex-col items-end">
      <span className="text-xs font-bold tracking-[0.3em] text-black mb-1 uppercase">Collaboration</span>
      <div className="w-32 h-1 bg-black rounded-full"></div>
    </div>
  </div>
);

const HeroSection = () => (
  <div className="bg-black rounded-[3rem] p-8 md:p-12 text-white shadow-2xl relative overflow-hidden min-h-[500px] flex flex-col md:flex-row gap-8">
    {/* Background Gradient Effect (Subtle) */}
    <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-gray-900 to-black -z-10"></div>

    {/* Left Content */}
    <div className="flex-1 flex flex-col justify-between relative z-10">
      <div>
        <div className="flex items-center gap-4 mb-6">
          <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-black font-bold text-lg">
            0.1
          </div>
          <div className="h-px bg-gray-600 w-48"></div>
          <span className="text-gray-400 text-xs uppercase tracking-wider">Creative Developer</span>
        </div>

        <h1 className="text-[5rem] md:text-[7rem] font-bold leading-none tracking-tighter mb-2 font-sans mix-blend-difference">
          PORTO
          <br />
          FOLIO
        </h1>
        
        {/* Subtitle Strip */}
        <div className="bg-white text-black px-4 py-1 inline-block rounded-sm mt-4 transform -rotate-1">
          <p className="text-sm font-bold uppercase tracking-widest">Luvi Asakura — Creative Developer & AI Enthusiast</p>
        </div>
      </div>

      {/* Action Buttons Bottom Left */}
      <div className="flex gap-4 mt-12">
        <button className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-black hover:scale-110 transition">
          <Heart fill="black" size={20} />
        </button>
        <button className="w-12 h-12 bg-gray-300 rounded-full flex items-center justify-center text-black hover:scale-110 transition">
          <Bookmark fill="black" size={20} />
        </button>
        <button className="w-12 h-12 bg-gray-300 rounded-full flex items-center justify-center text-black hover:scale-110 transition">
          <Share2 size={20} />
        </button>
      </div>
    </div>

    {/* Right Content (Floating Card Look) */}
    <div className="flex-1 relative flex flex-col justify-center md:pl-12">
       {/* Top Right Pill Buttons */}
       <div className="flex gap-4 justify-end mb-8">
          <button className="bg-[#1a1a1a] border border-gray-700 px-8 py-2 rounded-full text-xs font-bold tracking-wider hover:bg-white hover:text-black transition">VIEW CV</button>
          <button className="bg-[#1a1a1a] border border-gray-700 px-8 py-2 rounded-full text-xs font-bold tracking-wider hover:bg-white hover:text-black transition">HIRE ME</button>
       </div>

      {/* The White Card */}
      <div className="bg-white text-black rounded-[2.5rem] p-8 shadow-2xl relative">
        <h2 className="text-2xl font-bold mb-4">Siapa Luvi Asakura?</h2>
        <p className="text-gray-600 text-sm leading-relaxed mb-8">
          Selamat datang di portofolio saya. Saya seorang AI Engineer & Web Developer yang fokus pada desain antarmuka, automasi AI, dan pembuatan aplikasi kreatif. Menggabungkan estetika dengan logika kode.
        </p>
        
        {/* Custom Slider UI in Card */}
        <div className="w-full bg-black h-3 rounded-full mb-2 relative">
           <div className="absolute right-0 -top-1.5 w-6 h-6 bg-white border-4 border-black rounded-full"></div>
        </div>

        {/* View More Button Floating */}
        <div className="absolute -bottom-6 -right-6">
           <button className="bg-gray-200 text-black px-8 py-3 rounded-full font-bold text-sm shadow-lg flex items-center gap-2 hover:bg-white transition">
             View more <div className="bg-black text-white rounded-full p-1"><ArrowRight size={14}/></div>
           </button>
        </div>
      </div>
    </div>
  </div>
);

const TimelineItem = ({ num, title, content, icon: Icon }) => (
  <div className="p-6 md:p-8 flex flex-col h-full border-r border-gray-800 last:border-r-0 relative group hover:bg-[#0a0a0a] transition duration-300">
    <div className="flex justify-between items-start mb-6">
      <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-black font-bold text-lg shadow-lg z-10">
        {num}
      </div>
      {Icon && <Icon className="text-gray-600 group-hover:text-white transition" size={24} />}
    </div>
    <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
    <p className="text-gray-400 text-xs leading-relaxed">
      {content}
    </p>
  </div>
);

const TimelineSection = () => (
  <div className="bg-black rounded-[3rem] shadow-2xl overflow-hidden text-white grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 divide-gray-800 relative">
     {/* Row 1 */}
     <TimelineItem 
       num="1" 
       title="Skills" 
       icon={Code}
       content="Penguasaan mendalam pada JavaScript, Python, React, dan integrasi AI Model untuk solusi web modern." 
     />
     <TimelineItem 
       num="2" 
       title="Projects" 
       icon={LayoutTemplate}
       content="Berbagai proyek full-stack dari e-commerce, dashboard analitik, hingga aplikasi generatif AI." 
     />
     <TimelineItem 
       num="3" 
       title="Experience" 
       icon={Award}
       content="3+ tahun pengalaman profesional bekerja dengan startup teknologi dan agensi kreatif internasional." 
     />
     
     {/* Divider Line for Visual accuracy (Absolute centered line horizontally if needed, but grid handles it) */}

     {/* Row 2 - Conceptually styling borders to match grid */}
     <div className="md:col-span-3 grid grid-cols-1 md:grid-cols-3 border-t border-gray-800">
        <TimelineItem 
          num="4" 
          title="Tools" 
          icon={Cpu}
          content="VS Code, Figma, Docker, TensorFlow, Git, dan Adobe Creative Suite sebagai senjata utama." 
        />
        <TimelineItem 
          num="5" 
          title="Certificates" 
          icon={Award}
          content="Sertifikasi Google Cloud Professional, Meta Frontend Developer, dan DeepLearning.AI Specialization." 
        />
        <TimelineItem 
          num="6" 
          title="Contact" 
          icon={Mail}
          content="Terbuka untuk kolaborasi freelance atau full-time. Hubungi saya via email atau LinkedIn." 
        />
     </div>
  </div>
);

const CarouselCard = ({ title, type, colorClass }) => (
  <div className="bg-[#0f0f0f] rounded-[2rem] p-4 w-full md:w-1/3 flex-shrink-0 relative group cursor-pointer overflow-hidden h-64">
    <div className={`absolute top-0 right-0 w-24 h-24 rounded-bl-[5rem] ${colorClass} opacity-80 transition group-hover:scale-150 duration-500`}></div>
    
    {/* Content Placeholder */}
    <div className="h-full flex flex-col justify-end z-10 relative">
      <div className="bg-white w-12 h-12 rounded-xl mb-4 flex items-center justify-center shadow-lg">
         {type === 'web' && <LayoutTemplate size={24} className="text-black"/>}
         {type === 'ai' && <Cpu size={24} className="text-black"/>}
         {type === 'ui' && <Palette size={24} className="text-black"/>}
      </div>
      <h4 className="text-white font-bold text-xl">{title}</h4>
      <div className="w-full h-1 bg-gray-800 mt-4 rounded-full overflow-hidden">
        <div className={`h-full w-1/3 ${colorClass}`}></div>
      </div>
    </div>
  </div>
);

const CarouselSection = () => (
  <div className="flex items-center gap-4">
    <button className="w-12 h-12 bg-white rounded-full shadow-lg flex items-center justify-center hover:bg-gray-100 flex-shrink-0">
      <ChevronLeft size={24} />
    </button>

    <div className="flex gap-6 overflow-x-auto w-full pb-4 hide-scrollbar">
      <CarouselCard title="Project Website" type="web" colorClass="bg-blue-500" />
      <CarouselCard title="AI Automation" type="ai" colorClass="bg-purple-500" />
      <CarouselCard title="UI/UX Illustration" type="ui" colorClass="bg-yellow-400" />
    </div>

    <button className="w-12 h-12 bg-white rounded-full shadow-lg flex items-center justify-center hover:bg-gray-100 flex-shrink-0">
      <ChevronRight size={24} />
    </button>
  </div>
);

const Footer = () => (
  <div className="bg-black text-white rounded-full px-8 py-4 mt-8 flex justify-between items-center shadow-2xl relative">
    {/* Decorative Lines */}
    <div className="absolute top-2 bottom-2 left-32 right-32 border-t border-b border-gray-800 hidden md:block"></div>

    <span className="text-xs font-bold tracking-[0.2em] z-10">ABOUT ME</span>
    
    {/* Center Home Button Pop-out */}
    <div className="bg-white w-16 h-16 -mt-8 rounded-full border-8 border-[#f0f0f0] flex items-center justify-center shadow-xl z-20 cursor-pointer transform hover:-translate-y-1 transition">
      <Home className="text-black" size={24} />
    </div>

    <span className="text-xs font-bold tracking-[0.2em] z-10">CONTACT</span>
  </div>
);

// --- MAIN APP ---

function App() {
  return (
    <div className="min-h-screen bg-[#f0f0f0] text-black font-sans p-4 md:p-8 max-w-[1440px] mx-auto space-y-8 selection:bg-black selection:text-white pb-12">
      <Navbar />
      <SearchSection />
      <HeroSection />
      <TimelineSection />
      <CarouselSection />
      <Footer />
      
      {/* Special styling for scrollbar hiding inside App scope */}
      <style>{`
        .hide-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .hide-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </div>
  );
}

export default App;