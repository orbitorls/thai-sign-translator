import type { ModelInfo, TranslateResult } from "./api/client";
import type { ConsentState } from "./privacy/consentStorage";

export const MOCKUP_MODE = import.meta.env.VITE_MOCKUP_MODE !== "false";

export const MOCK_LIVE_INTERVAL_MS = 1500;
export const MOCK_CALIBRATION_MS = 600;
export const MOCK_MIN_LIVE_FRAMES = 8;

export const MOCK_MODELS: ModelInfo[] = [
  {
    id: "conductor_core",
    label_th: "Conductor Core",
    label_en: "Conductor Core",
    architecture: "Streaming translator",
    available: true,
    default: true,
  },
  {
    id: "conductor_flow",
    label_th: "Conductor Flow",
    label_en: "Conductor Flow",
    architecture: "Fast response",
    available: true,
    default: false,
  },
  {
    id: "conductor_scene",
    label_th: "Conductor Scene",
    label_en: "Conductor Scene",
    architecture: "Conversation model",
    available: true,
    default: false,
  },
];

const MOCK_WORDS = [
  "สวัสดี",
  "ครับ",
  "ค่ะ",
  "ขอบคุณ",
  "กินข้าว",
  "ดื่มน้ำ",
  "แมว",
  "หมา",
  "ฟ้า",
  "ฝนตก",
  "ร้อน",
  "เย็น",
  "เดิน",
  "วิ่ง",
  "นอน",
  "อ่านหนังสือ",
  "ดูทีวี",
  "โรงเรียน",
  "มหาวิทยาลัย",
  "ภาษามือ",
  "แปล",
  "ทดสอบ",
  "กล้อง",
  "มือ",
  "ยิ้ม",
  "หัวเราะ",
  "พรุ่งนี้",
  "วันนี้",
  "เมื่อวาน",
  "อร่อย",
  "หิว",
  "ง่วง",
  "สบายดี",
  "ไปไหน",
  "อยู่บ้าน",
  "ทำงาน",
  "พักผ่อน",
];

const MOCK_PHRASES = [
  "วันนี้อากาศดีมากเลยนะ",
  "อยากกินข้าวผัดปูมากๆ",
  "เดี๋ยวไปห้องสมุดก่อน",
  "ลองทำท่ามือดูสิ จะมีคำโผล่มา",
  "แมววิ่งไปทางซ้ายแล้วกลับมา",
  "ฝนตกแต่ก็ยังสนุกดี",
  "พรุ่งนี้มีสอบภาษามือ",
  "ขอบคุณที่ช่วยทดสอบ UI",
  "กล้องเปิดแล้ว ลองทำท่ามือดูนะ",
];

let mockSequence = 0;

function pickMockSentence(frames: number[][][]): string {
  mockSequence += 1;
  const seed = mockSequence * 17 + frames.length * 11 + Date.now();

  if (seed % 4 === 0) {
    return MOCK_PHRASES[seed % MOCK_PHRASES.length];
  }

  const wordCount = 2 + (seed % 5);
  const words: string[] = [];
  for (let i = 0; i < wordCount; i += 1) {
    const index = (seed + i * 29) % MOCK_WORDS.length;
    words.push(MOCK_WORDS[index]);
  }
  return words.join(" ");
}

export function createMockTranslateResult(
  frames: number[][][],
  modelId?: string,
): TranslateResult {
  const sentence = pickMockSentence(frames);
  const score = 0.55 + ((mockSequence % 38) / 100);
  return {
    sentence,
    score: Number(score.toFixed(2)),
    model: modelId ?? MOCK_MODELS[0].id,
    latency_ms: 90 + (mockSequence % 80),
    warning: null,
    token_score: Number((score - 0.05).toFixed(2)),
    landmark_quality: 0.72 + ((mockSequence % 20) / 100),
  };
}

export function createMockFrames(count = 24): number[][][] {
  return Array.from({ length: count }, (_, frameIndex) =>
    Array.from({ length: 8 }, (_, landmarkIndex) => {
      const base = frameIndex / Math.max(1, count - 1);
      const offset = landmarkIndex * 0.015;
      return [
        Number((0.25 + base * 0.4 + offset).toFixed(4)),
        Number((0.3 + offset).toFixed(4)),
        Number((offset / 2).toFixed(4)),
        0.98,
      ];
    }),
  );
}

export function createMockConsentState(): ConsentState {
  return {
    version: "2026-06-28-v1",
    updatedAt: new Date().toISOString(),
    modalComplete: true,
    scopes: {
      service: true,
      model_improvement: true,
      video_research: false,
      academic_publication: false,
    },
  };
}

export type MockSignCategory = "greetings" | "numbers" | "polite" | "basic";
export type MockSignLevel = "beginner" | "intermediate";

export interface MockSign {
  id: string;
  label_th: string;
  label_en: string;
  description_th: string;
  description_en: string;
  category: MockSignCategory;
  level: MockSignLevel;
}

export const MOCK_SIGNS_TH: MockSign[] = [
  {
    id: "sawasdee",
    label_th: "สวัสดี",
    label_en: "Hello",
    description_th: "ทักทายพื้นฐาน พับมือประนมไว้ที่ระดับอกพร้อมก้มศีรษะเล็กน้อย",
    description_en: "Basic greeting — palms together at chest level with a slight bow.",
    category: "greetings",
    level: "beginner",
  },
  {
    id: "khobkhun",
    label_th: "ขอบคุณ",
    label_en: "Thank you",
    description_th: "ใช้นิ้วโป้งแตะที่ปลายคาง แล้วยกมือออกไปข้างหน้า",
    description_en: "Thumb taps the chin, then hand moves forward.",
    category: "polite",
    level: "beginner",
  },
  {
    id: "khothot",
    label_th: "ขอโทษ",
    label_en: "Sorry",
    description_th: "มือขวาแตะที่หน้าอกเบา ๆ พร้อมพยักหน้าเล็กน้อย",
    description_en: "Right hand touches the chest lightly with a small nod.",
    category: "polite",
    level: "beginner",
  },
  {
    id: "chai",
    label_th: "ใช่",
    label_en: "Yes",
    description_th: "กำมือแล้วส่ายขึ้นลงเล็กน้อยเหมือนการพยักหน้า",
    description_en: "Closed fist nods up and down like a head nod.",
    category: "basic",
    level: "beginner",
  },
  {
    id: "mai",
    label_th: "ไม่",
    label_en: "No",
    description_th: "กางนิ้วชี้และนิ้วกลาง แล้วส่ายไปทางซ้ายขวา",
    description_en: "Index and middle finger extended, waving side to side.",
    category: "basic",
    level: "beginner",
  },
  {
    id: "neung",
    label_th: "หนึ่ง",
    label_en: "One",
    description_th: "ยกนิ้วโป้งขึ้น นิ้วที่เหลือกำ",
    description_en: "Thumb up, other fingers closed.",
    category: "numbers",
    level: "beginner",
  },
  {
    id: "song",
    label_th: "สอง",
    label_en: "Two",
    description_th: "ยกนิ้วโป้งและนิ้วชี้ขึ้น นิ้วที่เหลือกำ",
    description_en: "Thumb and index finger up, others closed.",
    category: "numbers",
    level: "beginner",
  },
  {
    id: "sam",
    label_th: "สาม",
    label_en: "Three",
    description_th: "ยกนิ้วโป้ง นิ้วชี้ และนิ้วกลางขึ้น",
    description_en: "Thumb, index, and middle finger up.",
    category: "numbers",
    level: "beginner",
  },
  {
    id: "sip",
    label_th: "สิบ",
    label_en: "Ten",
    description_th: "กำมือทั้งสองข้างประกบกัน แล้วหมุนเล็กน้อย",
    description_en: "Two closed fists tapped together, then rotated slightly.",
    category: "numbers",
    level: "intermediate",
  },
  {
    id: "kinkhao",
    label_th: "กินข้าว",
    label_en: "Eat",
    description_th: "นิ้วทั้งห้าของมือขวาประกบกัน แล้วนำเข้าปากเป็นจังหวะ",
    description_en: "Right fingertips bunch together and tap the mouth rhythmically.",
    category: "basic",
    level: "beginner",
  },
  {
    id: "nam",
    label_th: "น้ำ",
    label_en: "Water",
    description_th: "นิ้วโป้งและนิ้วกลางแตะกันเป็นรูปตัว W แล้วเคลื่อนมือลง",
    description_en: "Thumb and middle finger touch forming a W shape, hand moves down.",
    category: "basic",
    level: "beginner",
  },
  {
    id: "rak",
    label_th: "รัก",
    label_en: "Love",
    description_th: "กำมือทั้งสองไขว้กันที่ระดับอก แล้วกอดแน่น",
    description_en: "Two fists crossed at chest level, then hug tight.",
    category: "polite",
    level: "intermediate",
  },
];
