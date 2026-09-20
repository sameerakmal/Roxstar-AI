import type { Metadata } from "next";
import "@livekit/components-styles/prefabs/index.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoxStar — AI Voice Room",
  description:
    "Join a real-time voice room with Dost and Sathi, your AI companions by RoxStar.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
