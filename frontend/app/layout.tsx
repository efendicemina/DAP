import type { Metadata } from 'next'
import { ReactNode } from 'react'
import { Roboto_Flex } from 'next/font/google'
import '../styles/globals.css'

const robotoFlex = Roboto_Flex({ 
  subsets: ['latin', 'latin-ext'],
  display: 'swap',
  variable: '--font-roboto-flex',
})

export const metadata: Metadata = {
  title: 'MPR Chatbot - Ministarstvo Pravde BiH',
  description: 'NLP chatbot za pravnu podršku i digitalno usmjeravanje korisnika',
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: ReactNode
}) {
  return (
    <html lang="bs" className={`${robotoFlex.variable}`}>
      <body className="font-sans antialiased text-[16px] md:text-[18px]">
        {children}
      </body>
    </html>
  )
}
