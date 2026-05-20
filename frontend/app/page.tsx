'use client'

import { useState, useRef, useEffect } from 'react'
import { PaperAirplaneIcon, QuestionMarkCircleIcon, SparklesIcon, DocumentTextIcon, BuildingLibraryIcon, IdentificationIcon } from '@heroicons/react/24/outline'
import { ChatBubbleLeftRightIcon, AcademicCapIcon, MapPinIcon } from '@heroicons/react/24/solid'

interface Message {
  id: string
  text: string
  sender: 'user' | 'bot'
  timestamp: Date
  confidence?: number
  category?: string
}

interface ChatResponse {
  response: string
  confidence: number
  category: string
  source: string
  timestamp: string
  query: string
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      text: 'Zdravo! 👋 Ja sam MPR Chatbot - vaš asistent za pravne i administrativne informacije Ministarstva Pravde BiH. Mogu vam pomoći da pronađete zakone, pravilnike, obrasce i procedure. Kako vam mogu pomoći danas?',
      sender: 'bot',
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  useEffect(() => {
    const fetchSuggestedQuestions = async () => {
      try {
        const response = await fetch(`${API_URL}/suggested-questions`)
        const data = await response.json()
        setSuggestedQuestions(data)
      } catch (error) {
        console.error('Failed to fetch suggested questions:', error)
      }
    }
    fetchSuggestedQuestions()
  }, [API_URL])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (messageText: string) => {
    if (!messageText.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      sender: 'user',
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: messageText, language: 'bs' }),
      })

      if (!response.ok) throw new Error('Failed to get response')

      const data: ChatResponse = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        sender: 'bot',
        timestamp: new Date(),
        confidence: data.confidence,
        category: data.category,
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          text: 'Izvinite, došlo je do greške pri obradi vašeg pitanja. Molim pokušajte ponovno.',
          sender: 'bot',
          timestamp: new Date(),
        },
      ])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  return (
    <div className="flex h-screen bg-slate-50 font-sans selection:bg-blue-100">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-slate-200 hidden md:flex flex-col shadow-sm z-10 relative">
        <div className="p-8 border-b border-slate-100 bg-gradient-to-br from-blue-50 to-white">
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-blue-600 p-2 rounded-xl text-white shadow-lg shadow-blue-200">
              <BuildingLibraryIcon className="w-6 h-6" />
            </div>
            <h1 className="text-2xl font-extrabold tracking-tight text-slate-800">
              MPR
              <span className="text-blue-600">.bot</span>
            </h1>
          </div>
          <p className="text-slate-500 text-sm font-medium">Digitalni Asistent</p>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-8">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <SparklesIcon className="w-4 h-4 text-amber-400" /> Često traženo
          </h3>
          <div className="space-y-3">
            {suggestedQuestions.map((q, idx) => (
              <button
                key={idx}
                onClick={() => sendMessage(q)}
                className="w-full text-left p-4 rounded-2xl bg-white border border-slate-200 hover:border-blue-300 hover:shadow-md hover:-translate-y-0.5 hover:ring-1 hover:ring-blue-100 transition-all duration-200 text-slate-600 font-medium text-[15px] group flex gap-3"
              >
                <QuestionMarkCircleIcon className="w-5 h-5 text-slate-400 group-hover:text-blue-500 shrink-0" />
                <span className="leading-snug">{q}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="p-6 bg-slate-50 border-t border-slate-200">
          <div className="flex gap-3 bg-white p-4 rounded-2xl shadow-sm border border-slate-100">
            <IdentificationIcon className="w-8 h-8 text-blue-400 shrink-0" />
            <p className="text-xs text-slate-500 leading-relaxed">
              Ovaj sistem služi za informisanje bazirano na javno objavljenim podacima. Sistem <span className="font-semibold text-slate-700">ne pruža zvanične pravne savjete</span>.
            </p>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col relative bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-fixed">
        
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-md border-b border-slate-200 px-8 py-5 shadow-sm sticky top-0 z-20 flex justify-between items-center">
          <div className="flex flex-col">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              <ChatBubbleLeftRightIcon className="w-6 h-6 text-blue-600" /> Pravno Usmjeravanje
            </h2>
            <span className="text-sm font-medium text-slate-500 flex items-center gap-2 mt-1">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
              </span>
              Asistent je aktivan
            </span>
          </div>
          <div className="md:hidden">
            {/* Mobile icon placeholder */}
            <BuildingLibraryIcon className="w-8 h-8 text-blue-600" />
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 md:px-12 py-8 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex animate-slide-up ${
                message.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.sender === 'bot' && (
                <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center shrink-0 mr-4 shadow-md text-white">
                  <AcademicCapIcon className="w-6 h-6" />
                </div>
              )}
              
              <div
                className={`max-w-2xl px-6 py-4 relative group ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white rounded-3xl rounded-tr-md shadow-md shadow-blue-200'
                    : 'bg-white text-slate-700 rounded-3xl rounded-tl-md shadow-sm border border-slate-100 hover:shadow-md transition-shadow'
                }`}
              >
                <p className="text-base md:text-lg leading-relaxed whitespace-pre-wrap">{message.text}</p>
                
                <div className={`mt-3 flex items-center justify-between opacity-80 ${message.sender === 'user' ? 'text-blue-100' : 'text-slate-400'}`}>
                  <span className="text-xs font-medium">
                    {message.timestamp.toLocaleTimeString('bs-BA', { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start animate-fade-in">
              <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center shrink-0 mr-4 shadow-md text-white">
                 <AcademicCapIcon className="w-6 h-6" />
              </div>
              <div className="bg-white px-6 py-5 rounded-3xl rounded-tl-md shadow-sm border border-slate-100 flex items-center gap-2">
                <div className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-bounce"></div>
                <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                <div className="w-2.5 h-2.5 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} className="h-4" />
        </div>

        {/* Input Form */}
        <div className="p-4 md:p-8 bg-gradient-to-t from-slate-50 relative pb-8 md:pb-12">
          <div className="max-w-4xl mx-auto relative">
            <div className="relative bg-white shadow-xl shadow-slate-200/50 rounded-full border border-slate-200 transition-all focus-within:ring-4 focus-within:ring-blue-100 focus-within:border-blue-400 flex items-center p-2">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isLoading) {
                    e.preventDefault();
                    sendMessage(inputValue);
                  }
                }}
                placeholder="Pronađite zakone, procedure ili obrasce..."
                className="flex-1 w-full bg-transparent px-6 py-4 text-slate-700 outline-none text-base md:text-lg placeholder:text-slate-400"
                disabled={isLoading}
              />
              
              <button
                onClick={() => sendMessage(inputValue)}
                disabled={isLoading || !inputValue.trim()}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 transform active:scale-95 transition-all text-white p-4 rounded-full font-medium shadow-md shadow-blue-200 disabled:shadow-none flex items-center justify-center shrink-0 mr-1"
              >
                <PaperAirplaneIcon className="w-6 h-6 -ml-0.5" />
              </button>
            </div>
            
            <p className="text-center text-xs md:text-sm text-slate-400 mt-4 font-medium px-4 flex items-center justify-center gap-2">
              <DocumentTextIcon className="w-4 h-4" />
              Sistem koristi podatke sa zvanične baze Ministarstva pravde BiH. 
            </p>
          </div>
        </div>

      </div>
    </div>
  )
}
