import React, { useState } from 'react';
import { CURRICULUM } from './constants';
import { Lesson } from './types';
import Sidebar from './components/Sidebar';
import VideoPlayer from './components/VideoPlayer';
import CodeBlock from './components/CodeBlock';
import AITutor from './components/AITutor';
import { Clock, BarChart, Terminal, Cpu, Bot, Menu, BookOpen } from 'lucide-react';

export default function App() {
  const [currentLesson, setCurrentLesson] = useState<Lesson>(CURRICULUM[0].lessons[0]);
  const [isTutorOpen, setIsTutorOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Helper to find next lesson
  const handleNextLesson = () => {
    let found = false;
    for (const mod of CURRICULUM) {
      for (const les of mod.lessons) {
        if (found) {
          setCurrentLesson(les);
          window.scrollTo(0,0);
          return;
        }
        if (les.id === currentLesson.id) found = true;
      }
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Sidebar - Desktop */}
      <Sidebar 
        curriculum={CURRICULUM} 
        currentLesson={currentLesson} 
        onSelectLesson={(l) => {
          setCurrentLesson(l);
          setMobileMenuOpen(false);
          window.scrollTo(0,0);
        }} 
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden relative">
        {/* Mobile Header */}
        <div className="md:hidden p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950">
          <span className="font-bold text-indigo-400">LLM Mastery</span>
          <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="text-slate-400">
            <Menu />
          </button>
        </div>

        {/* Mobile Menu Overlay */}
        {mobileMenuOpen && (
           <div className="absolute inset-0 z-40 bg-slate-950/95 backdrop-blur-sm p-4 overflow-y-auto">
             <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold">Curriculum</h2>
                <button onClick={() => setMobileMenuOpen(false)}><Menu /></button>
             </div>
             {CURRICULUM.map(mod => (
               <div key={mod.id} className="mb-4">
                 <h3 className="text-slate-500 text-xs font-bold uppercase mb-2">{mod.title}</h3>
                 {mod.lessons.map(les => (
                   <button 
                     key={les.id}
                     onClick={() => { setCurrentLesson(les); setMobileMenuOpen(false); }}
                     className="block w-full text-left py-2 px-3 rounded text-slate-300 hover:bg-slate-800"
                   >
                     {les.title}
                   </button>
                 ))}
               </div>
             ))}
           </div>
        )}

        <main className="flex-1 overflow-y-auto p-4 md:p-8 max-w-7xl mx-auto w-full">
          
          {/* Header */}
          <div className="flex flex-col md:flex-row md:items-start justify-between mb-8 gap-4 border-b border-slate-800 pb-6">
            <div className="flex-1">
              <div className="flex items-center space-x-2 text-indigo-400 text-sm mb-2 font-mono uppercase tracking-wide">
                <span>{CURRICULUM.find(m => m.lessons.some(l => l.id === currentLesson.id))?.title.split(':')[0]}</span>
                <span>/</span>
                <span>Lesson {currentLesson.id}</span>
              </div>
              <h1 className="text-3xl font-bold text-white mb-4">{currentLesson.title}</h1>
              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
                {currentLesson.description}
              </p>
            </div>
            
            <div className="flex flex-col space-y-3 min-w-[200px]">
               <div className="flex items-center space-x-2 text-slate-400 bg-slate-900 px-4 py-2 rounded-lg border border-slate-800">
                  <Clock size={16} />
                  <span>{currentLesson.durationMinutes} minutes</span>
               </div>
               <div className={`flex items-center space-x-2 px-4 py-2 rounded-lg border ${
                 currentLesson.difficulty === 'Beginner' ? 'bg-emerald-950/30 border-emerald-900 text-emerald-400' :
                 currentLesson.difficulty === 'Intermediate' ? 'bg-amber-950/30 border-amber-900 text-amber-400' :
                 'bg-rose-950/30 border-rose-900 text-rose-400'
               }`}>
                  <BarChart size={16} />
                  <span>{currentLesson.difficulty}</span>
               </div>
               <button 
                onClick={() => setIsTutorOpen(true)}
                className="hidden md:flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg transition-colors font-medium shadow-lg shadow-indigo-900/20"
               >
                 <Bot size={18} />
                 <span>Ask AI Tutor</span>
               </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Left Column: Content */}
            <div className="lg:col-span-2 space-y-10">
              
              {/* Video Player */}
              <section>
                {currentLesson.resources.filter(r => r.type === 'video').map((video, idx) => (
                  <div key={idx} className="mb-6">
                    <VideoPlayer url={video.url} title={video.title} />
                    <div className="mt-3 flex items-center justify-between">
                       <span className="text-sm text-slate-500 flex items-center">
                          <span className="bg-red-900/30 text-red-400 text-xs px-2 py-0.5 rounded mr-2">VIDEO</span>
                          {video.author}
                       </span>
                    </div>
                  </div>
                ))}
              </section>

              {/* Concepts */}
              <section>
                <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Core Concepts</h3>
                <div className="flex flex-wrap gap-2">
                  {currentLesson.concepts.map((concept, i) => (
                    <span key={i} className="bg-slate-800 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-md text-sm hover:border-indigo-500/50 transition-colors cursor-default">
                      {concept}
                    </span>
                  ))}
                </div>
              </section>

              {/* Objectives */}
              <section className="bg-slate-900/40 rounded-xl p-6 border border-slate-800/50">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                  <Terminal className="mr-2 text-indigo-400" size={20} />
                  Learning Objectives
                </h3>
                <ul className="space-y-3">
                  {currentLesson.objectives.map((obj, i) => (
                    <li key={i} className="flex items-start text-slate-300">
                      <div className="mt-1.5 mr-3 w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                      <span className="leading-relaxed">{obj}</span>
                    </li>
                  ))}
                </ul>
              </section>

              {/* Code Snippets */}
              <section>
                <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                  <Cpu className="mr-2 text-indigo-400" size={20} />
                  Implementation Details
                </h3>
                {currentLesson.codeSnippets.length > 0 ? (
                  currentLesson.codeSnippets.map((snippet, i) => (
                    <CodeBlock key={i} language={snippet.language} code={snippet.code} title={snippet.title} />
                  ))
                ) : (
                  <div className="p-8 border border-dashed border-slate-800 rounded-xl bg-slate-900/20 text-slate-500 text-sm text-center">
                    No code snippets for this lesson. Focus on the conceptual frameworks.
                  </div>
                )}
              </section>

            </div>

            {/* Right Column: Exercises & Progress */}
            <div className="space-y-6">
              
              <div className="bg-gradient-to-br from-indigo-950/20 to-slate-900 border border-indigo-500/20 rounded-xl p-6 shadow-xl sticky top-4">
                <div className="flex items-center space-x-2 mb-4 text-indigo-400">
                  <BookOpen size={20} />
                  <h3 className="text-lg font-bold text-white">Hands-on Exercise</h3>
                </div>
                
                <div className="prose prose-invert prose-sm mb-6 text-slate-300 leading-relaxed">
                  <p>{currentLesson.exercise.description}</p>
                </div>
                
                <div className="bg-slate-950 rounded p-4 border border-slate-800 mb-6">
                  <span className="text-xs font-bold text-slate-500 uppercase block mb-2">Success Criteria / Output</span>
                  <p className="font-mono text-sm text-emerald-400 whitespace-pre-wrap">{currentLesson.exercise.expectedOutput}</p>
                </div>

                <div className="flex flex-col space-y-3">
                  <button 
                    onClick={handleNextLesson}
                    className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-indigo-900/20 hover:shadow-indigo-900/40 transform hover:-translate-y-0.5"
                  >
                    Complete Lesson
                  </button>
                  <button 
                     onClick={() => setIsTutorOpen(true)}
                     className="md:hidden w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
                  >
                    <Bot size={18} />
                    <span>Ask AI Tutor</span>
                  </button>
                </div>
              </div>

              {/* Additional Resources */}
              <div className="border-t border-slate-800 pt-6">
                 <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Required Reading</h4>
                 <ul className="space-y-4">
                   {currentLesson.resources.slice(1).map((res, i) => (
                     <li key={i} className="group">
                       <a href={res.url} target="_blank" rel="noreferrer" className="block p-3 rounded-lg hover:bg-slate-900 border border-transparent hover:border-slate-800 transition-all">
                         <div className="flex items-start justify-between">
                            <span className="text-sm font-medium text-blue-400 group-hover:text-blue-300">{res.title}</span>
                            <span className="text-[10px] uppercase text-slate-600 border border-slate-800 px-1.5 rounded">{res.type}</span>
                         </div>
                         <div className="text-xs text-slate-500 mt-1">{res.author}</div>
                       </a>
                     </li>
                   ))}
                   {currentLesson.resources.length <= 1 && <li className="text-sm text-slate-600 italic">No additional reading required.</li>}
                 </ul>
              </div>

            </div>
          </div>
        </main>
      </div>

      {/* AI Tutor Sidebar Overlay */}
      <AITutor 
        currentLesson={currentLesson} 
        isOpen={isTutorOpen} 
        onClose={() => setIsTutorOpen(false)} 
      />
    </div>
  );
}